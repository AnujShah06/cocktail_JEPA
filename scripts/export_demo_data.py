"""
export_demo_data.py -- one-shot export of everything the demo page needs.

The interactive demo (a self-contained HTML page) cannot run a JEPA
checkpoint in a browser, so all model outputs are PRE-COMPUTED here, once,
into a single demo_data.json the page is built around.

This is a one-off utility, not part of the training/eval pipeline.

What it exports, for the three demo panels:

  1. SCORING  -- a sample of real test recipes, each with its energy and
     a percentile rank among all test recipes (so the page can say
     "this cocktail is more coherent than 80% of real recipes").

  2. CORRUPTION -- for each sampled recipe, its perturbations (one per
     type) with energies, so the page can show "corrupt it -> energy
     rises" with REAL pre-computed numbers, not faked ones.

  3. GENERATION -- a few partial recipes completed by energy descent,
     each with the generated ingredients and the completed recipe's
     energy: "give the model a base spirit, it builds a coherent drink".

  4. RESULTS -- the #43 comparison table, copied straight from
     ablation_table.json (the page's honest results panel).

Run on the machine that has the reference checkpoint:
    uv run python scripts/export_demo_data.py \\
        --ckpt runs/jepa-06-final/jepa-06-s1.ckpt \\
        --table runs/jepa-06-final/ablation_table.json \\
        --out demo/demo_data.json
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack, load_recipes
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.energy.energy import energy_over_loader, recipe_energy
from cocktail_jepa.generate.generate import GenConfig, generate
from cocktail_jepa.train.checkpoint import load_checkpoint


def _recipe_ingredients(recipe: dict) -> list[dict]:
    """Compact ingredient list for the page: name + proportion."""
    return [
        {"name": i["ingredient"],
         "proportion": i.get("proportion")}
        for i in recipe["ingredients"]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export demo_data.json.")
    ap.add_argument("--ckpt", required=True,
                    help="reference JEPA checkpoint")
    ap.add_argument("--table", default=None,
                    help="ablation_table.json from #43 (optional)")
    ap.add_argument("--out", default="demo/demo_data.json")
    ap.add_argument("--n-recipes", type=int, default=60,
                    help="how many test recipes to sample for the demo")
    ap.add_argument("--n-generate", type=int, default=6,
                    help="how many generation examples to produce")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = CONFIG.paths
    device = CONFIG.device
    splits = paths.corpus / "splits"
    rng = random.Random(args.seed)

    vocab = Vocabulary.from_file(paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)
    model = load_checkpoint(args.ckpt, map_location=device)["model"]
    model.to(device).eval()
    print(f"loaded reference checkpoint: {args.ckpt}")

    # ---- 1+2. score ALL test recipes, then sample for the demo --------
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")

    test_ds = CocktailDataset(test, vocab, max_len=args.max_len,
                              n_frequencies=args.n_frequencies)
    pert_ds = CocktailDataset(perturbed, vocab, max_len=args.max_len + 2,
                              n_frequencies=args.n_frequencies)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)
    pert_loader = DataLoader(pert_ds, batch_size=128, shuffle=False,
                             collate_fn=_stack)

    print("scoring test recipes ...")
    test_e, test_ids = energy_over_loader(model, test_loader, device=device)
    print("scoring perturbations ...")
    pert_e, pert_ids = energy_over_loader(model, pert_loader, device=device)

    # energy -> percentile rank among all real test recipes (low energy =
    # high coherence -> high percentile)
    sorted_e = torch.sort(test_e).values
    def percentile(e: float) -> float:
        # fraction of real recipes with HIGHER energy (worse) than this
        rank = float((sorted_e > e).sum().item()) / len(sorted_e)
        return round(100.0 * rank, 1)

    energy_by_id = {rid: float(e) for rid, e in zip(test_ids, test_e)}
    # perturbations grouped by source recipe id
    pert_by_source: dict[str, list[dict]] = {}
    pert_energy = {id(r): None for r in pert_ds.recipes}
    for r, e in zip(pert_ds.recipes, pert_e):
        sid = r.get("source_id", "")
        pert_by_source.setdefault(sid, []).append(
            {"perturbation": r.get("perturbation", "?"),
             "ingredients": _recipe_ingredients(r),
             "energy": round(float(e), 4)})

    # sample recipes that actually HAVE perturbations, mix of coherence
    candidates = [r for r in test_ds.recipes
                  if r.get("recipe_id", "") in pert_by_source]
    rng.shuffle(candidates)
    sample = candidates[:args.n_recipes]

    scoring = []
    for r in sample:
        rid = r.get("recipe_id", "")
        e = energy_by_id.get(rid)
        if e is None:
            continue
        scoring.append({
            "recipe_id": rid,
            "name": r.get("name", "Untitled"),
            "ingredients": _recipe_ingredients(r),
            "energy": round(e, 4),
            "percentile": percentile(e),
            "perturbations": pert_by_source.get(rid, []),
        })
    print(f"  exported {len(scoring)} scored recipes with perturbations")

    # global energy range, for the page's gauge scaling
    energy_stats = {
        "real_min": round(float(test_e.min()), 4),
        "real_max": round(float(test_e.max()), 4),
        "real_mean": round(float(test_e.mean()), 4),
        "real_std": round(float(test_e.std()), 4),
        "pert_mean": round(float(pert_e.mean()), 4),
    }

    # ---- 3. generation examples ---------------------------------------
    # complete a few partial recipes from a fixed base spirit
    print("generating example recipes (energy descent) ...")
    gen_seeds = [
        (["gin"], 3), (["white rum"], 3), (["bourbon"], 3),
        (["tequila"], 3), (["vodka"], 3), (["cognac"], 3),
    ][:args.n_generate]
    generation = []
    for fixed, n_gen in gen_seeds:
        try:
            out = generate(model, vocab, fixed_ingredients=fixed,
                           n_generate=n_gen, cfg=GenConfig(restarts=4,
                                                           steps=120),
                           max_len=args.max_len, prop_dim=prop_dim,
                           device=device)
            generation.append({
                "base": fixed,
                "generated": out["generated"],
                "full_recipe": out["full_recipe"],
                "energy": round(float(out["energy"]), 4),
                "percentile": percentile(float(out["energy"])),
            })
        except Exception as exc:  # generation is finicky; skip a failure
            print(f"  [skip] generation from {fixed}: {exc}")
    print(f"  exported {len(generation)} generation examples")

    # ---- 4. results table ---------------------------------------------
    results = None
    if args.table:
        try:
            results = json.load(open(args.table, encoding="utf-8"))
            print(f"  included results table from {args.table}")
        except Exception as exc:
            print(f"  [skip] results table: {exc}")

    # ---- write --------------------------------------------------------
    blob = {
        "energy_stats": energy_stats,
        "scoring": scoring,
        "generation": generation,
        "results": results,
        "note": ("All energies are pre-computed by the trained JEPA "
                 "reference model; the page replays them."),
    }
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
