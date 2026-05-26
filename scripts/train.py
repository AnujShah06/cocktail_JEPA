"""
train.py -- Stage 3 training entry point.

Wires the corpus, the JEPA model, and the training loop together, logs
through the W&B-or-console logging layer, and writes checkpoints to
runs/<run_name>/.

Local smoke run (small, fast, just proves the loop works on mps/cpu):
    uv run python scripts/train.py --smoke

Full run (intended for a cloud GPU):
    uv run python scripts/train.py --run-name jepa-01 --epochs 30

The cloud box sets COCKTAIL_DATA_DIR, COCKTAIL_RUN_DIR, COCKTAIL_WANDB
via env vars -- see config.py. No code changes needed to move to cloud.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, JEPAMaskCollator, load_recipes
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.logging import get_logger
from cocktail_jepa.model.jepa import build_jepa
from cocktail_jepa.train.loop import TrainConfig, train


def _seed_everything(seed: int) -> None:
    """
    Seed every RNG that affects a training run, so a given --seed is
    fully reproducible.  This is what makes the #17 multi-seed study
    valid: each seed must deterministically fix (a) weight init, (b) the
    DataLoader shuffle order, and (c) the collator's mask sampling -- and
    a re-run of the same seed must reproduce the result exactly.

    Note: torch.manual_seed covers weight init and the default-generator
    draws; the DataLoader is additionally given an explicit generator in
    main(), and the collator its own seed, so all three variance sources
    trace back to this one number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the cocktail JEPA.")
    ap.add_argument("--run-name", default="jepa-dev")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    # Phase-2 hyperparameters. --sigreg-weight is the LeJEPA single
    # regularizer trade-off (#18); the lambda sweep varies this.
    # --proportion-aux-weight is the #13 auxiliary-loss weight, held
    # fixed across the sweep so the sweep isolates the lambda effect.
    ap.add_argument("--sigreg-weight", type=float, default=1.0)
    ap.add_argument("--proportion-aux-weight", type=float, default=0.5)
    # --no-ema is the #43 "without EMA" ablation: no separate momentum
    # target encoder; the context encoder is the target branch under a
    # stop-gradient.  Default keeps EMA on (the reference model).
    ap.add_argument("--no-ema", action="store_true",
                    help="#43 ablation: train without the EMA target encoder")
    # --seed fixes init, data-shuffle order, and mask sampling, so a run
    # is reproducible.  The #17 multi-seed study varies ONLY this.
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to verify the loop works")
    args = ap.parse_args()

    _seed_everything(args.seed)

    if args.smoke:
        args.epochs, args.batch_size, args.run_name = 2, 32, "smoke"

    paths = CONFIG.paths
    splits = paths.corpus / "splits"
    if not (splits / "train.jsonl").exists():
        print(f"[FAIL] no splits at {splits} -- run scripts/prepare_data.py first")
        return 1

    # data
    vocab = Vocabulary.from_file(paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)
    train_ds = CocktailDataset(load_recipes(splits / "train.jsonl"), vocab,
                               max_len=args.max_len,
                               n_frequencies=args.n_frequencies)
    val_ds = CocktailDataset(load_recipes(splits / "val.jsonl"), vocab,
                             max_len=args.max_len,
                             n_frequencies=args.n_frequencies)
    if args.smoke:  # shrink for a fast loop check
        train_ds.recipes = train_ds.recipes[:256]
        val_ds.recipes = val_ds.recipes[:128]

    # the DataLoader shuffle order is made deterministic per --seed via an
    # explicit generator; the collator's mask sampling likewise uses the
    # seed.  Together with _seed_everything's torch.manual_seed (init),
    # all three variance sources trace back to the one --seed.
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

    # model -- the coarse vocabulary (#4) is passed through so the
    # TokenEncoder builds its hierarchical coarse+fine embedding.  Omitting
    # these would silently construct a DEGENERATE single-coarse model and
    # #4 would contribute nothing.
    model = build_jepa(
        vocab_size=len(vocab), prop_dim=prop_dim,
        coarse_size=vocab.coarse_size,
        coarse_ids=vocab.coarse_ids,
        sigreg_weight=args.sigreg_weight,
        proportion_aux_weight=args.proportion_aux_weight,
        use_ema=not args.no_ema,
    )
    print(f"device: {CONFIG.device}")
    print(f"model parameters: {model.num_parameters()}")
    print(f"train recipes: {len(train_ds)}  val recipes: {len(val_ds)}")

    # logging
    logger = get_logger(
        run_name=args.run_name,
        config={"epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "device": CONFIG.device,
                "seed": args.seed,
                "use_ema": not args.no_ema,
                "sigreg_weight": args.sigreg_weight,
                "proportion_aux_weight": args.proportion_aux_weight,
                "model": model.num_parameters()},
        tags=["stage3", "smoke" if args.smoke else "full"],
    )

    # train
    run_dir = paths.runs / args.run_name
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    best = train(model, train_loader, val_loader, cfg,
                 device=CONFIG.device, logger=logger, run_dir=run_dir)

    logger.finish()
    print(f"\nStage 3 run complete. best checkpoint: {best}")
    print("  inspect the collapse diagnostics (effective_rank, mean_variance,")
    print("  embedding_spread) before trusting the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
