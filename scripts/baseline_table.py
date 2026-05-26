"""
baseline_table.py -- score the trivial #43 baselines and tabulate them.

Runs every trivial (non-learned) baseline from energy.baselines over the
test set and the perturbation set, computes the SAME overall + per-type
AUROC the JEPA is evaluated with, and prints a table.  These rows are the
trivial-baseline section of the #43 comparison table; the JEPA row and
the ablation / MAE / contrastive rows are added by later scripts.

This script trains nothing and loads no checkpoint -- the baselines have
no parameters.  It is fast and runs anywhere.

Run:
    uv run --no-sync python scripts/baseline_table.py
"""

from __future__ import annotations

import argparse
import json

import torch

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import load_recipes
from cocktail_jepa.data.vocab import Vocabulary
from cocktail_jepa.energy.baselines import (
    BASELINE_NAMES,
    build_log_rarity,
    score_recipes,
)
from cocktail_jepa.energy.evaluate import evaluate_energy


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the trivial baselines.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the 'random' baseline")
    args = ap.parse_args()

    paths = CONFIG.paths
    splits = paths.corpus / "splits"
    for needed in (splits / "test.jsonl", splits / "perturbations.jsonl",
                   paths.vocabulary):
        if not needed.exists():
            print(f"[FAIL] missing {needed}")
            return 1

    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    pert_types = [r.get("perturbation", "unknown") for r in perturbed]

    vocab = Vocabulary.from_file(paths.vocabulary)
    vocab_json = json.load(open(paths.vocabulary, encoding="utf-8"))
    log_rarity, default_rarity = build_log_rarity(vocab, vocab_json)

    print(f"test recipes: {len(test)}  perturbed: {len(perturbed)}")
    print(f"baselines: {BASELINE_NAMES}\n")

    # collect every baseline's report
    reports: dict[str, dict] = {}
    for name in BASELINE_NAMES:
        real_scores = score_recipes(test, name, log_rarity, default_rarity,
                                    seed=args.seed)
        pert_scores = score_recipes(perturbed, name, log_rarity,
                                    default_rarity, seed=args.seed)
        report = evaluate_energy(
            torch.tensor(real_scores, dtype=torch.float64),
            torch.tensor(pert_scores, dtype=torch.float64),
            pert_types,
        )
        reports[name] = report

    # ---- table ---------------------------------------------------------
    all_types = sorted(reports[BASELINE_NAMES[0]]["auroc_by_type"].keys())
    col_w = 20
    header = f"{'baseline':<{col_w}}{'overall':>10}"
    for t in all_types:
        header += f"{t[:11]:>13}"
    print(header)
    print("-" * len(header))
    for name in BASELINE_NAMES:
        r = reports[name]
        row = f"{name:<{col_w}}{r['auroc_overall']:>10.4f}"
        for t in all_types:
            row += f"{r['auroc_by_type'][t]:>13.4f}"
        print(row)

    print("\nReading the table:")
    print("  random ~ 0.50 confirms the harness is sound.")
    print("  length should be high on insert / incompatible_pair (those")
    print("    perturbations add an ingredient) and ~0.5 elsewhere.")
    print("  proportion_entropy should be high on over_dilution and ~0.5")
    print("    on scramble (a permutation leaves proportion entropy fixed).")
    print("  The trained JEPA must clearly beat these to justify the")
    print("  energy claim; per-type, it should beat them where they are")
    print("  strong, not only on average.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
