"""
perturb.py -- build the held-out perturbation set.

Stage 4 evaluates the energy function by asking: can it separate real
recipes from incoherent ones? That requires a set of KNOWN-bad recipes.
None exist as natural data, so we synthesize them by corrupting real
TEST recipes -- and only test recipes, so nothing here ever touches
training or model selection.

Three perturbation types, each a different kind of incoherence:
  * substitute  -- swap one ingredient for a random unrelated one
  * scramble    -- shuffle the proportions across slots (right ingredients,
                   wrong balance)
  * insert      -- add a random extra ingredient that does not belong

Each perturbed recipe is tagged with its type and its source recipe id,
so Stage 4 can report discrimination per perturbation type.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from cocktail_jepa.data.vocab import Vocabulary


def _all_ingredient_names(recipes: list[dict]) -> list[str]:
    """Pool of canonical ingredient names seen across the recipes."""
    names = set()
    for r in recipes:
        for i in r["ingredients"]:
            names.add(i["ingredient"])
    return sorted(names)


def perturb_substitute(recipe: dict, pool: list[str], rng: random.Random) -> dict:
    """Replace one ingredient with a random one not already in the recipe."""
    out = copy.deepcopy(recipe)
    present = {i["ingredient"] for i in out["ingredients"]}
    candidates = [n for n in pool if n not in present]
    if not candidates:
        return out
    slot = rng.randrange(len(out["ingredients"]))
    out["ingredients"][slot]["ingredient"] = rng.choice(candidates)
    out["ingredients"][slot]["category"] = "perturbed"
    return out


def perturb_scramble(recipe: dict, rng: random.Random) -> dict:
    """Keep the ingredients, shuffle the proportions across slots."""
    out = copy.deepcopy(recipe)
    props = [i.get("proportion") for i in out["ingredients"]]
    shuffled = props[:]
    # ensure it actually changes (for >=2 distinct values)
    for _ in range(8):
        rng.shuffle(shuffled)
        if shuffled != props:
            break
    for i, p in zip(out["ingredients"], shuffled):
        i["proportion"] = p
    return out


def perturb_insert(recipe: dict, pool: list[str], rng: random.Random) -> dict:
    """Add one extra random ingredient that is not already present."""
    out = copy.deepcopy(recipe)
    present = {i["ingredient"] for i in out["ingredients"]}
    candidates = [n for n in pool if n not in present]
    if not candidates:
        return out
    out["ingredients"].append({
        "ingredient": rng.choice(candidates),
        "qty_oz": None,
        "category": "perturbed",
        "proportion": None,
    })
    out["n_ingredients"] = len(out["ingredients"])
    return out


def make_perturbation_set(
    test_recipes: list[dict],
    seed: int = 42,
) -> list[dict]:
    """
    Build the perturbation set from test recipes.

    Each test recipe produces one perturbed copy per perturbation type.
    Returns a flat list; each item carries:
      perturbation : "substitute" | "scramble" | "insert"
      source_id    : recipe_id of the real recipe it was derived from
    """
    rng = random.Random(seed)
    pool = _all_ingredient_names(test_recipes)
    out: list[dict] = []

    for recipe in test_recipes:
        for kind, fn in (
            ("substitute", lambda r: perturb_substitute(r, pool, rng)),
            ("scramble", lambda r: perturb_scramble(r, rng)),
            ("insert", lambda r: perturb_insert(r, pool, rng)),
        ):
            p = fn(recipe)
            p["perturbation"] = kind
            p["source_id"] = recipe.get("recipe_id", "")
            out.append(p)

    return out


def write_perturbation_set(perturbed: list[dict], path: str | Path) -> Path:
    """Write the perturbation set to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in perturbed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
