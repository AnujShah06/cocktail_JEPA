"""
evaluate.py -- the headline experiment.

The question Stage 4 answers: does the energy function actually
distinguish coherent cocktails from incoherent ones?

Method: score the energy of every real TEST recipe and every PERTURBED
recipe (the corrupted test recipes built in Stage 1). A real recipe
should get LOW energy, a perturbed one HIGH energy. We measure how well
the energy separates the two populations with AUROC -- the probability
that a randomly chosen real recipe scores lower than a randomly chosen
perturbed one.

  AUROC ~ 1.0  -- the energy cleanly separates real from corrupted;
                  the JEPA genuinely learned mixological coherence.
  AUROC ~ 0.5  -- the energy is no better than chance; the model
                  learned nothing useful.

AUROC is also reported per perturbation type (substitute / scramble /
insert), since the model may catch some kinds of incoherence more
easily than others.

This module computes AUROC directly (no sklearn dependency) via the
rank-sum identity: AUROC = (sum of ranks of positives - n_pos terms) /
(n_pos * n_neg).
"""

from __future__ import annotations

import torch


def auroc(neg_scores: torch.Tensor, pos_scores: torch.Tensor) -> float:
    """
    AUROC where POSITIVE = perturbed (should score HIGH energy) and
    NEGATIVE = real (should score LOW energy).

    Computed via the Mann-Whitney U / rank-sum identity, so no external
    dependency. Ties are handled with average ranks.
    """
    neg = neg_scores.flatten()
    pos = pos_scores.flatten()
    n_neg, n_pos = neg.numel(), pos.numel()
    if n_neg == 0 or n_pos == 0:
        return float("nan")

    all_scores = torch.cat([neg, pos])
    # average ranks (1-indexed) to handle ties correctly
    order = torch.argsort(all_scores)
    ranks = torch.empty_like(all_scores)
    ranks[order] = torch.arange(1, all_scores.numel() + 1,
                                dtype=all_scores.dtype)
    # average-rank tie correction
    sorted_scores = all_scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while (j + 1 < sorted_scores.numel()
               and sorted_scores[j + 1] == sorted_scores[i]):
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1

    pos_rank_sum = ranks[n_neg:].sum().item()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def evaluate_energy(
    real_energies: torch.Tensor,
    perturbed_energies: torch.Tensor,
    perturbation_types: list[str],
) -> dict:
    """
    Compute the energy-discrimination report.

    real_energies        : [N_real]   energy of each real test recipe
    perturbed_energies   : [N_pert]   energy of each perturbed recipe
    perturbation_types   : [N_pert]   the kind tag per perturbed recipe

    Returns a dict with overall AUROC, per-type AUROC, and summary stats.
    """
    report: dict = {}

    # overall: real (negative) vs all perturbed (positive)
    report["auroc_overall"] = auroc(real_energies, perturbed_energies)

    # per perturbation type
    types = sorted(set(perturbation_types))
    type_tensor = perturbed_energies
    report["auroc_by_type"] = {}
    for t in types:
        mask = torch.tensor([pt == t for pt in perturbation_types])
        subset = type_tensor[mask]
        report["auroc_by_type"][t] = auroc(real_energies, subset)

    # descriptive stats -- useful for a sanity check and for the writeup
    report["stats"] = {
        "n_real": real_energies.numel(),
        "n_perturbed": perturbed_energies.numel(),
        "real_energy_mean": float(real_energies.mean()),
        "real_energy_std": float(real_energies.std()),
        "perturbed_energy_mean": float(perturbed_energies.mean()),
        "perturbed_energy_std": float(perturbed_energies.std()),
    }
    return report


def format_report(report: dict) -> str:
    """Pretty-print the evaluation report for the console."""
    s = report["stats"]
    lines = [
        "ENERGY DISCRIMINATION REPORT",
        "=" * 40,
        "",
        f"Overall AUROC (real vs. perturbed) : {report['auroc_overall']:.4f}",
        "",
        "AUROC by perturbation type:",
    ]
    for t, a in sorted(report["auroc_by_type"].items()):
        lines.append(f"  {t:12s} : {a:.4f}")
    lines += [
        "",
        f"real recipes      : {s['n_real']}",
        f"perturbed recipes : {s['n_perturbed']}",
        f"energy  real      : {s['real_energy_mean']:.4f} "
        f"+/- {s['real_energy_std']:.4f}",
        f"energy  perturbed : {s['perturbed_energy_mean']:.4f} "
        f"+/- {s['perturbed_energy_std']:.4f}",
        "",
        "Interpretation: AUROC near 1.0 means the energy cleanly separates",
        "real cocktails from corrupted ones. Near 0.5 means no signal.",
        "Perturbed energy should be HIGHER than real energy.",
    ]
    return "\n".join(lines)
