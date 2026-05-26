"""
ablation_table.py -- assemble the full #43 comparison table.

#43 asks: is the JEPA's latent-prediction objective actually responsible
for the energy function's quality?  This script answers it by scoring
every comparison point on the SAME energy AUROC metric and tabulating
them with bootstrap 95% confidence intervals.

ROWS
----
  JEPA (reference)   -- the locked Phase-2 model, one seed per --jepa-ckpt
                        (pass all 5 seed checkpoints to get mean +- std)
  - EMA              -- ablation: no momentum target encoder
  - regularizer      -- ablation: SIGReg off (sigreg_weight 0)
  MAE                -- baseline: masked-autoencoder objective
  contrastive        -- baseline: InfoNCE objective
  trivial baselines  -- random / length / rarity / proportion_entropy
                        (non-learned; from energy.baselines)

Each model row's energy uses that model's OWN energy function:
  JEPA / ablations -> energy.recipe_energy        (latent prediction error)
  MAE              -> mae.recipe_energy_mae       (reconstruction error)
  contrastive      -> contrastive.recipe_energy_contrastive (InfoNCE loss)
All three are "mean masked-slot error", so the AUROC numbers are
comparable; the trivial baselines are scored by their surface statistic.

Every row is bootstrapped (percentile method, default 10,000 resamples)
so the table reports AUROC with a 95% CI -- the #43 table and #22's CIs
in one place.  For the JEPA reference, multiple seed checkpoints are
combined as mean +- std ACROSS seeds (the #17 axis), and each seed is
also bootstrapped; the row shows both.

Run on the pod, after all checkpoints exist:
    uv run --no-sync python scripts/ablation_table.py \\
        --jepa-ckpt runs/jepa-06-s1/best.ckpt \\
        --jepa-ckpt runs/jepa-06-s2/best.ckpt \\
        --jepa-ckpt runs/jepa-06-s3/best.ckpt \\
        --jepa-ckpt runs/jepa-06-s4/best.ckpt \\
        --jepa-ckpt runs/jepa-06-s5/best.ckpt \\
        --noema-ckpt runs/jepa-noema/best.ckpt \\
        --noreg-ckpt runs/jepa-noreg/best.ckpt \\
        --mae-ckpt   runs/mae-01/best.ckpt \\
        --contrastive-ckpt runs/contrastive-01/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack, load_recipes
from cocktail_jepa.data.vocab import Vocabulary
from cocktail_jepa.energy.baselines import (
    BASELINE_NAMES,
    build_log_rarity,
    score_recipes,
)
from cocktail_jepa.energy.energy import energy_over_loader
from cocktail_jepa.energy.evaluate import auroc
from cocktail_jepa.energy.bootstrap import bootstrap_auroc_ci
from cocktail_jepa.train.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# energy computation per model family
# ---------------------------------------------------------------------------

def _loaders(vocab, splits, max_len, n_freq):
    """Build the test and perturbation DataLoaders (shared by all models)."""
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    test_ds = CocktailDataset(test, vocab, max_len=max_len,
                              n_frequencies=n_freq)
    # +2 length headroom: insert / incompatible_pair append an ingredient
    pert_ds = CocktailDataset(perturbed, vocab, max_len=max_len + 2,
                              n_frequencies=n_freq)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)
    pert_loader = DataLoader(pert_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)
    pert_types = [r.get("perturbation", "unknown") for r in pert_ds.recipes]
    return test_loader, pert_loader, pert_types


def _jepa_energies(ckpt, test_loader, pert_loader, device):
    """Real + perturbed energies for a JEPA / ablation checkpoint."""
    model = load_checkpoint(ckpt, map_location=device)["model"]
    model.to(device).eval()
    real, _ = energy_over_loader(model, test_loader, device=device)
    pert, _ = energy_over_loader(model, pert_loader, device=device)
    return real, pert


def _mae_energies(ckpt, test_loader, pert_loader, device):
    from cocktail_jepa.model.mae import (load_mae_checkpoint,
                                         mae_energy_over_loader)
    model = load_mae_checkpoint(ckpt, map_location=device)["model"]
    model.to(device).eval()
    real, _ = mae_energy_over_loader(model, test_loader, device=device)
    pert, _ = mae_energy_over_loader(model, pert_loader, device=device)
    return real, pert


def _contrastive_energies(ckpt, test_loader, pert_loader, device):
    from cocktail_jepa.model.contrastive import (
        contrastive_energy_over_loader, load_contrastive_checkpoint)
    model = load_contrastive_checkpoint(ckpt, map_location=device)["model"]
    model.to(device).eval()
    real, _ = contrastive_energy_over_loader(model, test_loader, device=device)
    pert, _ = contrastive_energy_over_loader(model, pert_loader, device=device)
    return real, pert


# ---------------------------------------------------------------------------
# one table row
# ---------------------------------------------------------------------------

def _row_from_energies(real, pert, pert_types, n_boot):
    """Bootstrap CI overall + per type for one (real, perturbed) pair."""
    overall = bootstrap_auroc_ci(real, pert, n_boot=n_boot)
    by_type = {}
    for t in sorted(set(pert_types)):
        mask = torch.tensor([pt == t for pt in pert_types])
        by_type[t] = bootstrap_auroc_ci(real, pert[mask], n_boot=n_boot)
    return {"overall": overall, "by_type": by_type}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the #43 comparison table.")
    ap.add_argument("--jepa-ckpt", action="append", default=[],
                    help="JEPA reference checkpoint(s); repeat for seeds")
    ap.add_argument("--noema-ckpt", default=None)
    ap.add_argument("--noreg-ckpt", default=None)
    ap.add_argument("--mae-ckpt", default=None)
    ap.add_argument("--contrastive-ckpt", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    paths = CONFIG.paths
    device = CONFIG.device
    splits = paths.corpus / "splits"
    vocab = Vocabulary.from_file(paths.vocabulary)
    vocab_json = json.load(open(paths.vocabulary, encoding="utf-8"))
    print(f"device: {device}  |  n_boot: {args.n_boot}")

    test_loader, pert_loader, pert_types = _loaders(
        vocab, splits, args.max_len, args.n_frequencies)
    all_types = sorted(set(pert_types))

    rows: list[tuple[str, dict]] = []          # (label, row-or-multiseed)

    # --- JEPA reference (possibly several seeds) ------------------------
    if args.jepa_ckpt:
        seed_rows = []
        for ck in args.jepa_ckpt:
            print(f"scoring JEPA {ck} ...")
            real, pert = _jepa_energies(ck, test_loader, pert_loader, device)
            seed_rows.append(_row_from_energies(real, pert, pert_types,
                                                args.n_boot))
        if len(seed_rows) == 1:
            rows.append(("JEPA (reference)", seed_rows[0]))
        else:
            # multi-seed: mean +- std across seeds for overall + each type
            pts = [r["overall"]["point"] for r in seed_rows]
            agg = {"overall": {"point": statistics.mean(pts),
                               "std": statistics.pstdev(pts),
                               "lo": min(pts), "hi": max(pts),
                               "multiseed": True, "n_seeds": len(pts)},
                   "by_type": {}}
            for t in all_types:
                tp = [r["by_type"][t]["point"] for r in seed_rows]
                agg["by_type"][t] = {"point": statistics.mean(tp),
                                     "std": statistics.pstdev(tp),
                                     "lo": min(tp), "hi": max(tp),
                                     "multiseed": True}
            rows.append((f"JEPA (reference, {len(pts)} seeds)", agg))

    # --- ablations ------------------------------------------------------
    if args.noema_ckpt:
        print(f"scoring -EMA {args.noema_ckpt} ...")
        real, pert = _jepa_energies(args.noema_ckpt, test_loader,
                                    pert_loader, device)
        rows.append(("  - EMA", _row_from_energies(real, pert, pert_types,
                                                   args.n_boot)))
    if args.noreg_ckpt:
        print(f"scoring -regularizer {args.noreg_ckpt} ...")
        real, pert = _jepa_energies(args.noreg_ckpt, test_loader,
                                    pert_loader, device)
        rows.append(("  - regularizer",
                     _row_from_energies(real, pert, pert_types, args.n_boot)))

    # --- SSL baselines --------------------------------------------------
    if args.mae_ckpt:
        print(f"scoring MAE {args.mae_ckpt} ...")
        real, pert = _mae_energies(args.mae_ckpt, test_loader,
                                   pert_loader, device)
        rows.append(("MAE baseline",
                     _row_from_energies(real, pert, pert_types, args.n_boot)))
    if args.contrastive_ckpt:
        print(f"scoring contrastive {args.contrastive_ckpt} ...")
        real, pert = _contrastive_energies(args.contrastive_ckpt,
                                           test_loader, pert_loader, device)
        rows.append(("contrastive baseline",
                     _row_from_energies(real, pert, pert_types, args.n_boot)))

    # --- trivial baselines ---------------------------------------------
    log_rarity, default_rarity = build_log_rarity(vocab, vocab_json)
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")
    for name in BASELINE_NAMES:
        real = torch.tensor(score_recipes(test, name, log_rarity,
                                          default_rarity),
                            dtype=torch.float64)
        pert = torch.tensor(score_recipes(perturbed, name, log_rarity,
                                          default_rarity),
                            dtype=torch.float64)
        rows.append((f"  trivial: {name}",
                     _row_from_energies(real, pert, pert_types, args.n_boot)))

    # ---- print the table ----------------------------------------------
    def fmt(cell: dict) -> str:
        if cell.get("multiseed"):
            return f"{cell['point']:.3f}+-{cell['std']:.3f}"
        return f"{cell['point']:.3f}[{cell['lo']:.2f},{cell['hi']:.2f}]"

    print("\n" + "=" * 78)
    print("#43 COMPARISON TABLE -- energy AUROC (95% CI, or mean+-std "
          "across seeds)")
    print("=" * 78)
    label_w = 26
    print(f"{'row':<{label_w}}{'overall':>20}")
    print("-" * (label_w + 20))
    for label, row in rows:
        print(f"{label:<{label_w}}{fmt(row['overall']):>20}")

    print("\nper-perturbation-type AUROC (point estimate):")
    header = f"{'row':<{label_w}}"
    for t in all_types:
        header += f"{t[:12]:>14}"
    print(header)
    print("-" * len(header))
    for label, row in rows:
        line = f"{label:<{label_w}}"
        for t in all_types:
            line += f"{row['by_type'][t]['point']:>14.3f}"
        print(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([{"row": l, **r} for l, r in rows], f, indent=1)
        print(f"\nwrote {args.out}")

    print("\nReading the table: the JEPA reference row should beat every")
    print("trivial baseline OVERALL; the ablation rows (- EMA, - regularizer)")
    print("show what each JEPA component contributes; the MAE and")
    print("contrastive rows show whether a different SSL objective on the")
    print("same encoder yields a comparable energy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
