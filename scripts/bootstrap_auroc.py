"""
bootstrap_auroc.py -- bootstrap confidence intervals for energy AUROC.

This is the tool for Phase-3 fix #22 (bootstrap AUROC CIs).  It is used
first, before Phase 3 proper, to settle one question: the aux-weight
sweep gave overall AUROC 0.683 / 0.677 / 0.687 across three checkpoints
-- is the 0.687-vs-0.683 gap a real difference or just noise?  A single
AUROC number cannot answer that; a confidence interval can.

METHOD -- the percentile bootstrap.
  Given the real-recipe energies and the perturbed-recipe energies for a
  checkpoint, resample BOTH populations WITH REPLACEMENT (same sizes),
  recompute AUROC on each resample, and repeat n_boot times (default
  10,000).  The 2.5th and 97.5th percentiles of that distribution are a
  95% confidence interval for the true AUROC.

  Resampling reals and perturbed independently is the standard two-sample
  bootstrap for a rank statistic; it captures the sampling variability of
  AUROC given finite test/perturbation sets.

  Note this measures variability from the FINITE EVALUATION SET only -- it
  does NOT capture training-seed variability.  That is a separate axis,
  handled by the multi-seed runs (#17): #17 varies the seed, #22 (this)
  varies the resample.  The two are reported together in Phase 3.

If several checkpoints are passed, their CIs are printed side by side and
the script states whether the best two OVERLAP -- the actual decision
input for the reference-checkpoint choice.

Run on the pod (pass every checkpoint to compare):
    uv run --no-sync python scripts/bootstrap_auroc.py \\
        --ckpt runs/jepa-05-sig05/best.ckpt \\
        --ckpt runs/jepa-05-aux10/best.ckpt \\
        --ckpt runs/jepa-05-aux20/best.ckpt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack, load_recipes
from cocktail_jepa.data.vocab import Vocabulary
from cocktail_jepa.energy.energy import energy_over_loader
from cocktail_jepa.energy.evaluate import auroc
from cocktail_jepa.train.checkpoint import load_checkpoint
from cocktail_jepa.energy.bootstrap import bootstrap_auroc_ci  # core of #22


def _energies_for(ckpt_path: str, vocab, splits, device,
                   max_len: int, n_freq: int) -> tuple[torch.Tensor,
                                                       torch.Tensor,
                                                       list[str]]:
    """Real + perturbed energies and per-perturbation tags for a checkpoint."""
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = ck["model"]
    model.to(device).eval()

    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    test_ds = CocktailDataset(test, vocab, max_len=max_len,
                              n_frequencies=n_freq)
    # +2 length headroom: insert / incompatible_pair APPEND an ingredient
    # (matches evaluate.py) so no perturbed recipe is silently dropped.
    pert_ds = CocktailDataset(perturbed, vocab, max_len=max_len + 2,
                              n_frequencies=n_freq)

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)
    pert_loader = DataLoader(pert_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)
    real_e, _ = energy_over_loader(model, test_loader, device=device)
    pert_e, _ = energy_over_loader(model, pert_loader, device=device)
    pert_types = [r.get("perturbation", "unknown") for r in pert_ds.recipes]
    return real_e, pert_e, pert_types


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap AUROC CIs.")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path; repeat to compare several")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    ap.add_argument("--per-type", action="store_true",
                    help="also bootstrap each perturbation type")
    args = ap.parse_args()

    paths = CONFIG.paths
    device = CONFIG.device
    splits = paths.corpus / "splits"
    vocab = Vocabulary.from_file(paths.vocabulary)
    print(f"device: {device}  |  n_boot: {args.n_boot}")

    results = []
    for ckpt in args.ckpt:
        print(f"\nscoring {ckpt} ...")
        real_e, pert_e, pert_types = _energies_for(
            ckpt, vocab, splits, device, args.max_len, args.n_frequencies)
        overall = bootstrap_auroc_ci(real_e, pert_e, n_boot=args.n_boot)
        entry = {"ckpt": ckpt, "overall": overall}
        if args.per_type:
            entry["by_type"] = {}
            for t in sorted(set(pert_types)):
                mask = torch.tensor([pt == t for pt in pert_types])
                entry["by_type"][t] = bootstrap_auroc_ci(
                    real_e, pert_e[mask], n_boot=args.n_boot)
        results.append(entry)

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 60)
    print("BOOTSTRAP AUROC -- 95% percentile CIs")
    print("=" * 60)
    for e in results:
        o = e["overall"]
        print(f"\n{e['ckpt']}")
        print(f"  overall AUROC : {o['point']:.4f}  "
              f"95% CI [{o['lo']:.4f}, {o['hi']:.4f}]")
        if "by_type" in e:
            for t, c in e["by_type"].items():
                print(f"    {t:18s}: {c['point']:.4f}  "
                      f"[{c['lo']:.4f}, {c['hi']:.4f}]")

    # ---- overlap verdict for the best two ------------------------------
    if len(results) >= 2:
        ranked = sorted(results, key=lambda e: e["overall"]["point"],
                        reverse=True)
        a, b = ranked[0], ranked[1]
        ao, bo = a["overall"], b["overall"]
        overlap = ao["lo"] <= bo["hi"] and bo["lo"] <= ao["hi"]
        print("\n" + "-" * 60)
        print(f"best:        {a['ckpt']}  "
              f"{ao['point']:.4f} [{ao['lo']:.4f}, {ao['hi']:.4f}]")
        print(f"runner-up:   {b['ckpt']}  "
              f"{bo['point']:.4f} [{bo['lo']:.4f}, {bo['hi']:.4f}]")
        if overlap:
            print("VERDICT: CIs OVERLAP -- the gap is within evaluation "
                  "noise; the two are not distinguishable on overall AUROC.")
        else:
            print("VERDICT: CIs are DISJOINT -- the gap is unlikely to be "
                  "evaluation noise; the best checkpoint is a real winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
