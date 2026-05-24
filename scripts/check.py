"""
check.py -- workbench smoke test.

Run this right after setup, and any time you move to a new machine
(e.g. the first time you SSH into a cloud GPU). It verifies:
  1. the cocktail_jepa package imports
  2. torch is present and which device is available
  3. the corpus is in place and readable

Usage:
    uv run python scripts/check.py
"""

import json
import sys


def main() -> int:
    ok = True

    # 1. package import -------------------------------------------------
    try:
        import cocktail_jepa
        from cocktail_jepa.config import CONFIG
        print(f"[ok]   cocktail_jepa v{cocktail_jepa.__version__} imports")
    except Exception as e:
        print(f"[FAIL] cannot import cocktail_jepa: {e}")
        return 1  # nothing else will work

    # 2. torch + device -------------------------------------------------
    try:
        import torch
        print(f"[ok]   torch {torch.__version__}")
        print(f"[info] selected device: {CONFIG.device}")
        if CONFIG.device == "cpu":
            print("[warn] no GPU/MPS -- fine for Stage 1, too slow for training")
        # a real tensor op on the chosen device
        x = torch.randn(8, 8, device=CONFIG.device)
        _ = (x @ x).sum().item()
        print(f"[ok]   tensor op succeeds on {CONFIG.device}")
    except Exception as e:
        print(f"[FAIL] torch problem: {e}")
        ok = False

    # 3. logging layer -------------------------------------------------
    try:
        from cocktail_jepa.logging import get_logger
        # build a logger in console mode -- verifies the no-op path works
        lg = get_logger(run_name="station-check", use_wandb=False)
        lg.log({"smoke_test": 1.0}, step=0)
        lg.finish()
        print(f"[ok]   logging layer works (W&B {'on' if CONFIG.use_wandb else 'off'})")
    except Exception as e:
        print(f"[FAIL] logging layer problem: {e}")
        ok = False

    # 4. corpus ---------------------------------------------------------
    recipes = CONFIG.paths.recipes
    vocab = CONFIG.paths.vocabulary
    if recipes.exists():
        n = sum(1 for _ in open(recipes, encoding="utf-8"))
        print(f"[ok]   corpus: {n} recipes at {recipes}")
    else:
        print(f"[warn] no corpus at {recipes}")
        print("       drop recipes.jsonl + vocabulary.json into corpus/")
        ok = False
    if vocab.exists():
        v = json.load(open(vocab, encoding="utf-8"))
        print(f"[ok]   vocabulary: {v.get('size', '?')} ingredients")

    print()
    print("STATION READY" if ok else "STATION INCOMPLETE -- see [FAIL]/[warn] above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
