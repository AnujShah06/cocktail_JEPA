"""
transfer.py -- Stage 6 transfer experiment entry point.

Runs the headline transfer comparison: a classification head trained on
the FROZEN pretrained JEPA encoder vs. the same head on a FROM-SCRATCH
encoder, across several label fractions. A positive, label-scarce-biased
gap is the evidence the self-supervised representation transfers.

    uv run python scripts/transfer.py --ckpt runs/jepa-04/best.ckpt

Runs on CPU/MPS -- only the small head trains, the encoders are frozen.
"""

from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, load_recipes, _stack
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.model.jepa import build_jepa
from cocktail_jepa.train.checkpoint import load_checkpoint
from cocktail_jepa.transfer.labels import (
    SPIRIT_CLASSES, base_spirit, recipe_label,
)
from cocktail_jepa.transfer.sft import (
    format_transfer_report, run_transfer_comparison,
)


def _spirit_slot(recipe: dict) -> int:
    """Index of the base-spirit ingredient within the recipe."""
    sp = base_spirit(recipe)
    for i, ing in enumerate(recipe["ingredients"]):
        if ing["ingredient"] == sp:
            return i
    return 0


def _make_batches(recipes, vocab, prop_dim, max_len, batch_size, device):
    """Build labelled batches: each adds `label` and `spirit_index`."""
    ds = CocktailDataset(recipes, vocab, max_len=max_len)
    # align labels + spirit slots with ds.recipes (CocktailDataset may
    # have dropped over-long recipes)
    labels, slots = [], []
    for r in ds.recipes:
        labels.append(recipe_label(r))
        slots.append(_spirit_slot(r))

    batches = []
    for start in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        batch = _stack(items)
        idxs = list(range(start, min(start + batch_size, len(ds))))
        batch["label"] = torch.tensor([labels[i] for i in idxs])
        batch["spirit_index"] = torch.tensor([slots[i] for i in idxs])
        batches.append(batch)
    return batches


def main() -> int:
    ap = argparse.ArgumentParser(description="JEPA transfer experiment.")
    ap.add_argument("--ckpt", default="runs/jepa-04/best.ckpt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    args = ap.parse_args()

    device = CONFIG.device
    print(f"device: {device}")

    paths = CONFIG.paths
    vocab = Vocabulary.from_file(paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)

    # pretrained encoder (frozen) from the checkpoint
    ck = load_checkpoint(args.ckpt, map_location=device)
    pretrained = ck["model"]
    print(f"loaded checkpoint: step {ck['step']}, "
          f"val_loss {ck['extra'].get('val_loss', 'n/a')}")

    # from-scratch encoder: same architecture, random weights
    from_scratch = build_jepa(vocab_size=len(vocab), prop_dim=prop_dim)

    # labelled data -- only recipes in the 5 spirit classes
    splits = paths.corpus / "splits"
    train_recipes = [r for r in load_recipes(splits / "train.jsonl")
                     if recipe_label(r) is not None]
    val_recipes = [r for r in load_recipes(splits / "val.jsonl")
                   if recipe_label(r) is not None]
    random.Random(CONFIG.seed).shuffle(train_recipes)
    print(f"labelled recipes: {len(train_recipes)} train, "
          f"{len(val_recipes)} val  ({len(SPIRIT_CLASSES)} classes)")

    train_batches = _make_batches(train_recipes, vocab, prop_dim,
                                  args.max_len, args.batch_size, device)
    val_batches = _make_batches(val_recipes, vocab, prop_dim,
                                args.max_len, args.batch_size, device)

    print("running transfer comparison (pretrained vs from-scratch) ...\n")
    results = run_transfer_comparison(
        pretrained, from_scratch, train_batches, val_batches,
        device=device, epochs=args.epochs,
    )
    print(format_transfer_report(results, len(SPIRIT_CLASSES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
