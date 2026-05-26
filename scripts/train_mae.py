"""
train_mae.py -- train the MAE baseline for the #43 comparison table.

The MAE baseline (model/mae.py) is the masked-autoencoder counterpart of
the JEPA: same TokenEncoder, same SetEncoder backbone, same training
budget -- but it RECONSTRUCTS the masked ingredient instead of predicting
its latent.  This script trains it.

It uses its OWN short training loop rather than train.loop.train().  That
loop is specific to the JEPA -- it calls model.ema_update(), and its
evaluate() reads JEPA-only loss keys (sigreg_term, proportion_aux) and
the JEPA's `context_encoder` attribute.  The MAE has none of those.
Rather than branch the JEPA's verified loop, the MAE gets a small
self-contained loop here: simpler, and it cannot regress JEPA training.

To keep the comparison fair, the budget matches train.py's defaults:
same epochs, batch size, lr, AdamW + cosine-warmup schedule, grad clip.

Run (after build_corpus.py + prepare_data.py):
    uv run --no-sync python scripts/train_mae.py --run-name mae-01 --epochs 80
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, JEPAMaskCollator, load_recipes
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.logging import get_logger
from cocktail_jepa.model.mae import build_mae
from cocktail_jepa.train.checkpoint import save_checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay -- matches train.loop's schedule."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def _evaluate(model, loader, device: str) -> dict:
    """Mean MAE loss components over a loader."""
    model.eval()
    sums = {"loss": 0.0, "ingredient_loss": 0.0, "proportion_loss": 0.0}
    n = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        for k in sums:
            sums[k] += float(out[k].item())
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in sums.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the MAE baseline.")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    ap.add_argument("--proportion-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    _seed_everything(args.seed)
    device = CONFIG.device
    paths = CONFIG.paths
    splits = paths.corpus / "splits"

    vocab = Vocabulary.from_file(paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)
    train_recipes = load_recipes(splits / "train.jsonl")
    val_recipes = load_recipes(splits / "val.jsonl")
    if args.smoke:
        train_recipes = train_recipes[:256]
        val_recipes = val_recipes[:128]

    train_ds = CocktailDataset(train_recipes, vocab, max_len=args.max_len,
                               n_frequencies=args.n_frequencies)
    val_ds = CocktailDataset(val_recipes, vocab, max_len=args.max_len,
                             n_frequencies=args.n_frequencies)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=JEPAMaskCollator(deterministic=False, seed=args.seed),
        drop_last=True, generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=JEPAMaskCollator(deterministic=True),
    )

    model = build_mae(
        vocab_size=len(vocab), prop_dim=prop_dim,
        coarse_size=vocab.coarse_size, coarse_ids=vocab.coarse_ids,
        proportion_weight=args.proportion_weight,
    ).to(device)
    print(f"device: {device}")
    print(f"model parameters: {model.num_parameters()}")
    print(f"train recipes: {len(train_ds)}  val recipes: {len(val_ds)}")

    epochs = 2 if args.smoke else args.epochs
    logger = get_logger(
        run_name=args.run_name,
        config={"epochs": epochs, "batch_size": args.batch_size,
                "lr": args.lr, "device": device, "seed": args.seed,
                "model_type": "mae",
                "model": model.num_parameters()},
        tags=["stage3", "baseline", "mae",
              "smoke" if args.smoke else "full"],
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_warmup(s, args.warmup_steps, total_steps),
    )

    run_dir = paths.runs / (args.run_name or "mae")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = run_dir / "best.ckpt"
    step = 0
    prev_periodic = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
            step += 1

        train_loss = epoch_loss / max(1, len(train_loader))
        val = _evaluate(model, val_loader, device)
        logger.log(step=step, epoch=epoch,
                   train_loss=train_loss, lr=scheduler.get_last_lr()[0],
                   **{f"val_{k}": v for k, v in val.items()})

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(best_path, model, optimizer, scheduler, step=step,
                            extra={"val_loss": val["loss"]})

        # periodic checkpoint -- keep only the latest (bounds disk use,
        # same policy as train.loop)
        if (epoch + 1) % 5 == 0:
            new_periodic = run_dir / f"epoch_{epoch+1}.ckpt"
            save_checkpoint(new_periodic, model, optimizer, scheduler,
                            step=step)
            if prev_periodic is not None and prev_periodic.exists():
                prev_periodic.unlink()
            prev_periodic = new_periodic

    logger.summary(best_val_loss=best_val, total_steps=step)
    print(f"\nMAE baseline training complete. best: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
