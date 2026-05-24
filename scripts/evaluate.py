"""
evaluate.py -- Stage 4 evaluation entry point.

Loads a trained JEPA checkpoint, computes the energy of every real test
recipe and every perturbed recipe, and reports how well the energy
separates them (AUROC). This is the headline experiment: it tells you
whether the model genuinely learned mixological coherence.

Run after Stage 3 has produced a checkpoint:
    uv run python scripts/evaluate.py --ckpt runs/jepa-01/best.ckpt

Runs on CPU/MPS -- it is inference only, no training, fast on a laptop.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, load_recipes, _stack
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.energy.energy import energy_over_loader
from cocktail_jepa.energy.evaluate import evaluate_energy, format_report
from cocktail_jepa.train.checkpoint import load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the JEPA energy.")
    ap.add_argument("--ckpt", default="runs/jepa-01/best.ckpt")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    args = ap.parse_args()

    paths = CONFIG.paths
    splits = paths.corpus / "splits"
    ckpt_path = paths.root / args.ckpt if not args.ckpt.startswith("/") \
        else args.ckpt
    for needed in (ckpt_path, splits / "test.jsonl",
                   splits / "perturbations.jsonl"):
        if not str(needed).startswith("/") and not needed.exists():
            print(f"[FAIL] missing {needed}")
            return 1

    device = CONFIG.device
    print(f"device: {device}")

    # load the trained model
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = ck["model"]
    print(f"loaded checkpoint: step {ck['step']}, "
          f"val_loss {ck['extra'].get('val_loss', 'n/a')}")

    # data
    vocab = Vocabulary.from_file(paths.vocabulary)
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    test_ds = CocktailDataset(test, vocab, max_len=args.max_len,
                              n_frequencies=args.n_frequencies)
    pert_ds = CocktailDataset(perturbed, vocab, max_len=args.max_len,
                              n_frequencies=args.n_frequencies)
    print(f"test recipes: {len(test_ds)}  perturbed recipes: {len(pert_ds)}")

    # plain collate -- the energy function supplies its own deterministic masks
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=_stack)
    pert_loader = DataLoader(pert_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=_stack)

    # compute energies
    print("scoring real test recipes ...")
    real_energy, _ = energy_over_loader(model, test_loader, device=device)
    print("scoring perturbed recipes ...")
    pert_energy, _ = energy_over_loader(model, pert_loader, device=device)

    # perturbation-type tags, aligned with pert_ds order
    pert_types = [r.get("perturbation", "unknown") for r in pert_ds.recipes]

    # the headline experiment
    report = evaluate_energy(real_energy, pert_energy, pert_types)
    print()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
