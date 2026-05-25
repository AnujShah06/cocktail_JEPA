"""
demo.py -- the cocktail JEPA demo.

Ties the project together in one interactive tool. You type ingredients;
the demo:
  1. resolves each typed name to its closest canonical vocabulary name
     (so 'bourbon' -> 'whiskey', 'lime' -> 'lime juice');
  2. scores the energy of what you have so far (coherence);
  3. completes the recipe by energy descent (Stage 5);
  4. scores the completed recipe's energy.

Two modes:
    # one-shot: give ingredients and a count on the command line
    uv run python scripts/demo.py --ckpt runs/jepa-04/best.ckpt \\
        --have bourbon lime --generate 2

    # interactive: omit --have, type recipes at the prompt
    uv run python scripts/demo.py --ckpt runs/jepa-04/best.ckpt

Runs on CPU/MPS. No training, no cloud.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.energy.energy import recipe_energy
from cocktail_jepa.generate.generate import GenConfig, generate
from cocktail_jepa.generate.resolve import resolve_all
from cocktail_jepa.train.checkpoint import load_checkpoint

REAL_ENERGY_REF = 0.32   # Stage 4: real recipes ~0.29, perturbed ~0.32


def _score_partial(model, vocab, names, prop_dim, max_len, device):
    """Energy of a (complete or partial) recipe given canonical names."""
    if len(names) < 2:
        return None  # energy needs >=2 slots to mask over
    fake = {"recipe_id": "", "n_ingredients": len(names),
            "ingredients": [{"ingredient": n, "proportion": 1.0 / len(names)}
                            for n in names]}
    ds = CocktailDataset([fake], vocab, max_len=max_len)
    if len(ds) == 0:
        return None
    batch = _stack([ds[0]])
    return float(recipe_energy(model, batch, device=device)[0])


def _resolve_and_report(raw_names, vocab):
    """Resolve typed names; print what each became; return canonical list."""
    resolutions = resolve_all(raw_names, vocab)
    canonical = []
    for r in resolutions:
        canonical.append(r.matched)
        if r.exact:
            print(f"  '{r.query}' -> {r.matched}")
        else:
            print(f"  '{r.query}' -> {r.matched}  "
                  f"(closest match, confidence {r.confidence})")
    return canonical


def _run_one(model, vocab, raw_have, n_generate, prop_dim, max_len, device):
    """One full demo cycle: resolve, score, generate, score."""
    print("\nresolving ingredients:")
    have = _resolve_and_report(raw_have, vocab)

    e_partial = _score_partial(model, vocab, have, prop_dim, max_len, device)
    if e_partial is not None:
        print(f"\nenergy of what you have : {e_partial:.4f}")

    if n_generate > 0:
        print(f"\ngenerating {n_generate} more ingredient(s) by energy "
              f"descent ...")
        cfg = GenConfig(restarts=6, steps=160)
        result = generate(model, vocab, have, n_generate, cfg=cfg,
                          max_len=max_len, prop_dim=prop_dim, device=device)
        print("\n" + "=" * 44)
        print("COMPLETED RECIPE")
        print("=" * 44)
        for name in result["full_recipe"]:
            tag = "(given)" if name in have else "(generated)"
            print(f"  {name:32s} {tag}")
        print()
        e = result["energy"]
        print(f"recipe energy : {e:.4f}   "
              f"(real cocktails ~0.29-0.32)")
        if e < REAL_ENERGY_REF:
            print("-> in the real-recipe range: a coherent recipe.")
        else:
            print("-> elevated: only loosely coherent.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cocktail JEPA demo.")
    ap.add_argument("--ckpt", default="runs/jepa-04/best.ckpt")
    ap.add_argument("--have", nargs="*", default=None,
                    help="ingredients you have (one-shot mode)")
    ap.add_argument("--generate", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    args = ap.parse_args()

    device = CONFIG.device
    print(f"device: {device}")
    ck = load_checkpoint(args.ckpt, map_location=device)
    model = ck["model"]
    model.to(device).eval()
    vocab = Vocabulary.from_file(CONFIG.paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)
    print(f"loaded checkpoint: step {ck['step']}")

    if args.have is not None:
        # one-shot mode
        _run_one(model, vocab, args.have, args.generate,
                 prop_dim, args.max_len, device)
        return 0

    # interactive mode
    print("\nCocktail JEPA demo -- interactive mode")
    print("type ingredients separated by commas, e.g.:  bourbon, lime")
    print("then it completes the recipe. Ctrl-C or empty line to quit.\n")
    while True:
        try:
            line = input("ingredients> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        raw = [x.strip() for x in line.split(",") if x.strip()]
        if not raw:
            continue
        _run_one(model, vocab, raw, args.generate,
                 prop_dim, args.max_len, device)
        print()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
