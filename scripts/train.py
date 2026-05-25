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

from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, JEPAMaskCollator, load_recipes
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.logging import get_logger
from cocktail_jepa.model.jepa import build_jepa
from cocktail_jepa.train.loop import TrainConfig, train


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
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to verify the loop works")
    args = ap.parse_args()

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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=JEPAMaskCollator(deterministic=False, seed=CONFIG.seed),
        drop_last=True,
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
    )
    print(f"device: {CONFIG.device}")
    print(f"model parameters: {model.num_parameters()}")
    print(f"train recipes: {len(train_ds)}  val recipes: {len(val_ds)}")

    # logging
    logger = get_logger(
        run_name=args.run_name,
        config={"epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "device": CONFIG.device,
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
