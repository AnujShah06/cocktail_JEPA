"""
splits.py -- leakage-controlled train/val/test partitioning.

The risk this guards against: a recipe and a near-identical variant of it
landing on opposite sides of the train/test boundary. If that happens,
test performance is inflated -- the model effectively saw the test recipe
during training.

Strategy: group recipes by an ingredient-set signature (the set of
canonical ingredient ids, order-independent). All recipes sharing a
signature go into the SAME split as one unit. Splitting happens at the
group level, never the recipe level, so no variant can straddle the
boundary.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from cocktail_jepa.data.vocab import Vocabulary


def _signature(recipe: dict, vocab: Vocabulary) -> tuple[int, ...]:
    """Order-independent ingredient-set signature for a recipe."""
    ids = sorted({vocab.encode(i["ingredient"]) for i in recipe["ingredients"]})
    return tuple(ids)


def make_splits(
    recipes: list[dict],
    vocab: Vocabulary,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Partition recipes into train/val/test with no ingredient-set leakage.

    Returns {"train": [...], "val": [...], "test": [...]}.
    Recipes sharing an ingredient-set signature are kept together.
    """
    # group recipes by signature
    groups: dict[tuple, list[dict]] = {}
    for r in recipes:
        groups.setdefault(_signature(r, vocab), []).append(r)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    # walk groups, assigning whole groups until each split hits its quota
    n_total = len(recipes)
    n_val_target = int(n_total * val_frac)
    n_test_target = int(n_total * test_frac)

    split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for key in group_keys:
        members = groups[key]
        if len(split["test"]) < n_test_target:
            split["test"].extend(members)
        elif len(split["val"]) < n_val_target:
            split["val"].extend(members)
        else:
            split["train"].extend(members)

    return split


def write_splits(
    split: dict[str, list[dict]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write each split to out_dir/{name}.jsonl. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, recs in split.items():
        p = out_dir / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = p
    return paths


def load_split(path: str | Path) -> list[dict]:
    """Read one split's .jsonl back into a list of recipe dicts."""
    return [json.loads(line) for line in open(path, encoding="utf-8")]
