"""
prepare_data.py -- Stage 1 data preparation entry point.

Reads the corpus, builds leakage-controlled train/val/test splits and the
held-out perturbation set, and writes them to corpus/splits/.

Run once after the corpus is in place:
    uv run python scripts/prepare_data.py

Outputs (corpus/splits/):
    train.jsonl  val.jsonl  test.jsonl   leakage-controlled splits
    perturbations.jsonl                  corrupted test recipes for Stage 4
"""

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import load_recipes
from cocktail_jepa.data.perturb import make_perturbation_set, write_perturbation_set
from cocktail_jepa.data.splits import make_splits, write_splits
from cocktail_jepa.data.vocab import Vocabulary


def main() -> int:
    paths = CONFIG.paths

    if not paths.recipes.exists():
        print(f"[FAIL] no corpus at {paths.recipes}")
        return 1

    print("loading corpus ...")
    recipes = load_recipes(paths.recipes)
    vocab = Vocabulary.from_file(paths.vocabulary)
    print(f"  {len(recipes)} recipes, {vocab.n_ingredients} ingredients")

    print("building leakage-controlled splits ...")
    split = make_splits(recipes, vocab, seed=CONFIG.seed)
    for name, recs in split.items():
        print(f"  {name:5s}: {len(recs)} recipes")

    out_dir = paths.corpus / "splits"
    split_paths = write_splits(split, out_dir)
    print(f"  written to {out_dir}/")

    print("building perturbation set from test recipes ...")
    perturbed = make_perturbation_set(split["test"], seed=CONFIG.seed)
    pert_path = write_perturbation_set(perturbed, out_dir / "perturbations.jsonl")
    by_kind: dict[str, int] = {}
    for p in perturbed:
        by_kind[p["perturbation"]] = by_kind.get(p["perturbation"], 0) + 1
    print(f"  {len(perturbed)} perturbed recipes: {by_kind}")
    print(f"  written to {pert_path}")

    print("\nStage 1 data prep complete.")
    print("  next: inspect with  uv run marimo edit notebooks/01_explore_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
