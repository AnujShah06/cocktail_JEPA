"""
generate.py -- Stage 5 generation entry point.

Completes a partial cocktail recipe by energy descent: you supply the
ingredients you want, the model invents the rest by minimizing energy.

Examples:
    # give it gin and lime juice, ask for 2 more ingredients
    uv run python scripts/generate.py --ckpt runs/jepa-04/best.ckpt \\
        --have "gin" "lime juice" --generate 2

    # build a 4-ingredient recipe from a single fixed spirit
    uv run python scripts/generate.py --ckpt runs/jepa-04/best.ckpt \\
        --have "bourbon whiskey" --generate 3

Runs on CPU/MPS -- no training, no cloud.

A reference point for reading the energy: in the Stage 4 evaluation,
real test recipes averaged ~0.29 energy and perturbed recipes ~0.32. A
generated recipe whose energy lands near the real-recipe range is the
Stage 5 success criterion.
"""

from __future__ import annotations

import argparse

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.generate.generate import GenConfig, generate
from cocktail_jepa.train.checkpoint import load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a cocktail by energy descent.")
    ap.add_argument("--ckpt", default="runs/jepa-04/best.ckpt")
    ap.add_argument("--have", nargs="*", default=[],
                    help="ingredient names to keep fixed")
    ap.add_argument("--generate", type=int, default=2,
                    help="how many new ingredients to invent")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    args = ap.parse_args()

    device = CONFIG.device
    print(f"device: {device}")

    # load the frozen trained model
    ck_path = args.ckpt
    ck = load_checkpoint(ck_path, map_location=device)
    model = ck["model"]
    print(f"loaded checkpoint: step {ck['step']}, "
          f"val_loss {ck['extra'].get('val_loss', 'n/a')}")

    vocab = Vocabulary.from_file(CONFIG.paths.vocabulary)
    prop_dim = proportion_encoding_dim(args.n_frequencies)

    # validate the fixed ingredients are in vocabulary
    unknown = [n for n in args.have if n not in vocab.token_to_id]
    if unknown:
        print(f"[warn] not in vocabulary, will be treated loosely: {unknown}")
        print("       check spelling against corpus/vocabulary.json")

    if not args.have and args.generate < 2:
        print("[FAIL] give at least one --have ingredient or --generate >= 2")
        return 1

    print(f"\nfixed ingredients : {args.have or '(none)'}")
    print(f"generating        : {args.generate} new ingredient(s)")
    print(f"energy descent    : {args.restarts} restarts x {args.steps} steps")
    print("running ...\n")

    cfg = GenConfig(steps=args.steps, restarts=args.restarts)
    result = generate(model, vocab, args.have, args.generate, cfg=cfg,
                      max_len=args.max_len, prop_dim=prop_dim, device=device)

    print("=" * 44)
    print("GENERATED RECIPE")
    print("=" * 44)
    for name in result["full_recipe"]:
        tag = "  (given)" if name in args.have else "  (generated)"
        print(f"  {name}{tag}")
    print()
    print(f"recipe energy        : {result['energy']:.4f}")
    print(f"  (real recipes ~0.29, perturbed ~0.32 in Stage 4 eval)")
    per = result["energy_per_restart"]
    print(f"energy per restart   : "
          f"min {min(per):.4f}  max {max(per):.4f}  "
          f"({len(per)} restarts)")
    print()
    if result["energy"] < 0.32:
        print("-> energy is in the real-recipe range: a coherent completion.")
    else:
        print("-> energy is elevated: the completion is only loosely coherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
