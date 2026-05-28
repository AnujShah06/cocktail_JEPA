"""
pmi_table.py -- fit and score the PMI co-occurrence baseline.

The competent-simple-learned rung of the #43 ladder.  Fits a pairwise-PMI
energy on the TRAIN split, scores the held-out test recipes and their
perturbations with the SAME evaluate_energy / AUROC machinery as the JEPA,
and prints the overall + per-type AUROC so the row drops straight into the
comparison table.

Unlike baseline_table.py (whose baselines have no parameters), this one
learns from data -- so it loads the train split to fit.  It still uses no
GPU and no checkpoint; fitting is dict arithmetic and runs in seconds.

Run:
    uv run --no-sync python scripts/pmi_table.py
    uv run --no-sync python scripts/pmi_table.py --smoothing 0.5
"""

from __future__ import annotations

import argparse

import torch

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import load_recipes
from cocktail_jepa.energy.evaluate import evaluate_energy, format_report
from cocktail_jepa.energy.pmi_baseline import PMIEnergy


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit + score the PMI baseline.")
    ap.add_argument("--smoothing", type=float, default=0.5,
                    help="add-k smoothing on pair counts (k)")
    args = ap.parse_args()

    paths = CONFIG.paths
    splits = paths.corpus / "splits"
    for needed in (splits / "train.jsonl", splits / "test.jsonl",
                   splits / "perturbations.jsonl"):
        if not needed.exists():
            print(f"[FAIL] missing {needed}")
            return 1

    train = load_recipes(splits / "train.jsonl")
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    pert_types = [r.get("perturbation", "unknown") for r in perturbed]

    print(f"fitting PMI on {len(train)} train recipes "
          f"(add-k smoothing k={args.smoothing}) ...")
    pmi = PMIEnergy.fit(train, smoothing=args.smoothing)
    # diagnostic: floor PMI for two median-frequency ingredients gives a sense
    # of where unseen-pair scores land for typical ingredients.
    sorted_p = sorted(pmi._unigram.values())
    median_count = sorted_p[len(sorted_p) // 2] if sorted_p else 1
    import math as _m
    median_p = median_count / pmi._n_recipes
    floor_for_median_pair = _m.log(pmi._p_ab_floor / (median_p * median_p))
    print(f"  learned PMI for {len(pmi._pair_pmi)} co-occurring pairs; "
          f"unseen-pair floor PMI for two median-frequency ingredients "
          f"= {floor_for_median_pair:.3f}\n")

    real_scores = pmi.score_recipes(test)
    pert_scores = pmi.score_recipes(perturbed)

    report = evaluate_energy(
        torch.tensor(real_scores, dtype=torch.float64),
        torch.tensor(pert_scores, dtype=torch.float64),
        pert_types,
    )

    print(format_report(report))
    print("\nRow for the #43 table:")
    print(f"  PMI co-occurrence   overall AUROC = "
          f"{report['auroc_overall']:.4f}")
    print("\nReading this row:")
    print("  This is the missing rung between the trivial heuristics and")
    print("  the deep SSL models.  If PMI lands near the JEPA (0.707), the")
    print("  deep model's interaction signal is largely captured by simple")
    print("  pairwise co-occurrence.  If PMI lands well below it, the deep")
    print("  model earns its parameters.  Either result is reportable; the")
    print("  point is to KNOW, not to assert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
