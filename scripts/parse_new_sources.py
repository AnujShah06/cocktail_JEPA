"""
parse_new_sources.py -- parse Death & Co. and TheCocktailDB into the
cocktail-jepa corpus schema, for the Phase-1 #2 data merge.

Both sources are converted to the same record shape build_corpus.py
emits:
  {"recipe_id", "name", "source", "category", "ingredients": [
       {"ingredient": <raw name>, "proportion": <float 0..1>}, ...]}

Quantities are normalized to a within-recipe proportion (each
ingredient's oz / total oz). Ingredient NAMES are left raw here --
the corpus build step applies the canonical normalization.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

UPLOADS = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/new_sources.jsonl")

# ---- quantity parsing -------------------------------------------------
_FRAC = {"1/2": .5, "1/4": .25, "3/4": .75, "1/3": .333, "2/3": .667,
         "1/8": .125, "1 1/2": 1.5, "1 1/4": 1.25, "1 3/4": 1.75,
         "2 1/2": 2.5}


def parse_oz(text: str) -> float | None:
    """Pull an ounce quantity out of free text; None if not parseable."""
    if not text:
        return None
    t = text.strip().lower()
    for frac, val in sorted(_FRAC.items(), key=lambda x: -len(x[0])):
        if t.startswith(frac):
            return val
    m = re.match(r"(\d+\.?\d*)", t)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _finalize(ings: list[tuple[str, float | None]]) -> list[dict]:
    """Turn [(name, oz)] into [{ingredient, proportion}] summing to 1."""
    known = [(n, q) for n, q in ings if q and q > 0]
    total = sum(q for _, q in known)
    out = []
    for name, q in ings:
        if q and q > 0 and total > 0:
            prop = q / total
        else:
            prop = None
        out.append({"ingredient": name.strip(), "proportion": prop})
    return out


# ---- Death & Co. ------------------------------------------------------
def parse_death_and_co() -> list[dict]:
    rows = list(csv.reader(open(UPLOADS / "death-and-co-raw-data.csv",
                                encoding="utf-8", errors="ignore")))
    recipes, current, name = [], [], None
    for r in rows:
        if len(r) < 6:
            continue
        recipe_name = r[2].strip()
        ingredient = r[3].strip()
        qty = r[5].strip()
        if not recipe_name and not ingredient:        # blank separator
            if current and name:
                recipes.append((name, current))
            current, name = [], None
            continue
        if recipe_name and recipe_name.upper() != "CLASSIC AN VINTAGE":
            name = recipe_name.title()
            if ingredient:
                current.append((ingredient, parse_oz(qty)))
    if current and name:
        recipes.append((name, current))

    out = []
    for i, (nm, ings) in enumerate(recipes):
        if not ings:
            continue
        out.append({
            "recipe_id": f"deathco_{i}", "name": nm,
            "source": "death_and_co", "category": None,
            "ingredients": _finalize(ings),
        })
    return out


# ---- TheCocktailDB ----------------------------------------------------
def parse_cocktaildb() -> list[dict]:
    names = {}
    for r in csv.DictReader(open(UPLOADS / "drinks.csv",
                                 encoding="utf-8", errors="ignore")):
        names[r["id"].strip()] = r["name"].strip()

    by_id: dict[str, list] = {}
    for r in csv.reader(open(UPLOADS / "ingredients.csv",
                             encoding="utf-8", errors="ignore")):
        if len(r) < 2 or r[0] == "id":
            continue
        rid, raw = r[0].strip(), r[1].strip()
        # raw looks like "1 oz  Coconut rum" -- split qty from name
        m = re.match(r"([\d/.\s]+?)\s*oz\s+(.*)", raw, re.IGNORECASE)
        if m:
            qty, ing = parse_oz(m.group(1)), m.group(2).strip()
        else:
            qty, ing = None, raw
        by_id.setdefault(rid, []).append((ing, qty))

    out = []
    for rid, ings in by_id.items():
        if rid not in names or not ings:
            continue
        out.append({
            "recipe_id": f"cocktaildb_{rid}", "name": names[rid],
            "source": "thecocktaildb", "category": None,
            "ingredients": _finalize(ings),
        })
    return out


def main() -> None:
    dc = parse_death_and_co()
    cdb = parse_cocktaildb()
    all_new = dc + cdb
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in all_new:
            f.write(json.dumps(rec) + "\n")
    print(f"Death & Co.   : {len(dc)} recipes")
    print(f"TheCocktailDB : {len(cdb)} recipes")
    print(f"total new     : {len(all_new)} -> {OUT}")
    # quick quality check
    qok = sum(1 for r in all_new
              for i in r["ingredients"] if i["proportion"] is not None)
    qtot = sum(len(r["ingredients"]) for r in all_new)
    print(f"quantity parse rate: {qok}/{qtot} = {qok/max(1,qtot):.1%}")


if __name__ == "__main__":
    main()
