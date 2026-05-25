"""
build_corpus.py -- canonicalize the corpus and build the hierarchical
vocabulary.  Phase-1 fix #4.

This script does two things, in order:

  1. CANONICALIZE every raw ingredient string into a clean "fine" token.
     The merged 7,420-recipe corpus carries ~2,300 distinct raw strings:
     uppercased Death & Co. brand names, un-stripped quantity prefixes
     from TheCocktailDB ("1 shot Dark rum"), and ordinary case/spacing
     noise.  Canonicalization strips leading quantities/units, lowercases,
     and folds recognizable brands down to their GENERIC fine token
     (e.g. "BUFFALO TRACE BOURBON" -> "bourbon", "ST-GERMAIN" ->
     "elderflower liqueur").

  2. ASSIGN every fine token a COARSE token.  The coarse level is a single
     table mixing ingredient CATEGORY and, for spirits only, SPIRIT FAMILY:
       - a non-spirit ingredient's coarse token is its category
         ("citrus" juice -> "juice", "simple syrup" -> "sweetener", ...)
       - a spirit ingredient's coarse token is its spirit family, so the
         whiskey subtypes (bourbon / scotch / rye / irish) stay DISTINCT
         fine tokens but share the single coarse token "whiskey".
     This is the granularity fix: jepa-04 collapsed bourbon and scotch;
     the hierarchical token  embedding(coarse) + embedding(fine)  lets the
     model see "these are both whiskeys" (shared coarse) without losing
     "bourbon != scotch" (distinct fine).

Anything the rules cannot confidently place lands in the [JUNK] coarse
bucket -- it is kept as a distinct fine token (never dropped: dropping it
would silently turn a recipe slot into a phantom [MASK]), but its coarse
token degrades to "unknown".  The [JUNK] occurrence rate is printed at the
end as an honest coverage number.

Output: corpus/vocabulary.json, schema (extends the old one additively):
    {
      "size":         N_fine + 2,          # incl. [PAD],[MASK]
      "coarse_size":  N_coarse + 2,
      "coarse_vocab": ["[PAD]","[MASK]","spirit","juice", ... ,"[JUNK]"],
      "ingredients": [
         {"name","count","category","coarse","coarse_id"}, ...
      ]
    }
The old loader read only "size" + "ingredients[].name", so this file still
loads anywhere the old one did; the new fields are purely additive.

Run:  uv run python scripts/build_corpus.py
(reads corpus/recipes.jsonl, writes corpus/vocabulary.json + ingredients.csv)
"""

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

from cocktail_jepa.config import CONFIG

# ---------------------------------------------------------------------------
# 1. CANONICALIZATION  (raw ingredient string -> clean fine token)
# ---------------------------------------------------------------------------

# units/measure words that can lead a raw string; stripped along with any
# numeric/fraction quantity in front of them.
_UNITS = (
    r"(oz|cl|ml|l|cup|cups|shot|shots|part|parts|dash|dashes|tsp|tbsp|tblsp|"
    r"tablespoon|teaspoon|inch|bottle|bottles|can|cans|chunk|chunks|sprig|"
    r"sprigs|slice|slices|wedge|wedges|leaf|leaves|cube|cubes|drop|drops|"
    r"pinch|piece|pieces|scoop|scoops|barspoon|splash|glass|quart|peel|twist|"
    r"sticks?|bunch|handful|mini|beaten)"
)
_QTY_RE = re.compile(
    r"^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*|a|one|two|half)?\s*"
    + _UNITS
    + r"?\s+",
    re.IGNORECASE,
)


def _strip_quantity(s: str) -> str:
    """Repeatedly peel a leading quantity/unit token off a raw string."""
    prev = None
    while prev != s:
        prev = s
        s = _QTY_RE.sub("", s).strip()
    return re.sub(r"\s+", " ", s).strip()


# Brand / keyword -> generic fine token.  Checked in order: the FIRST tuple
# whose any-keyword matches wins, so more specific entries come first
# (whiskey subtypes before the generic "whiskey" catch-all).  Folding a
# brand to its generic keeps the fine vocabulary learnable -- e.g. eight
# different bourbon brands all become the single fine token "bourbon".
_KEYWORD_GENERIC: list[tuple[tuple[str, ...], str]] = [
    # --- whiskey family: distinct generic fine tokens, all coarse "whiskey"
    (("bourbon",), "bourbon"),
    (("scotch", "islay", "speyside", "highland park", "laphroaig",
      "macallan", "bowmore", "caol ila", "bruichladdich"), "scotch"),
    (("irish whiskey", "irish whisky", "redbreast", "bushmills",
      "knappogue", "clontarf"), "irish whiskey"),
    (("rye",), "rye whiskey"),
    (("tennessee",), "tennessee whiskey"),
    (("wheat whiskey", "wheat whisky", "bernheim"), "wheat whiskey"),
    (("whisky", "whiskey"), "whiskey"),
    # --- other spirits: generic fine token == its own name
    (("mezcal",), "mezcal"),
    (("tequila",), "tequila"),
    (("genever",), "genever"),
    (("gin",), "gin"),
    (("vodka",), "vodka"),
    (("cachaca", "cachaça"), "cachaca"),
    (("pisco",), "pisco"),
    (("cognac",), "cognac"),
    (("armagnac",), "armagnac"),
    (("calvados", "apple brandy", "applejack"), "apple brandy"),
    (("brandy",), "brandy"),
    (("absinthe", "pontarlier"), "absinthe"),
    (("rhum", "rum"), "rum"),
    (("aquavit", "akvavit"), "aquavit"),
    # --- a few common branded liqueurs -> generic liqueur fine tokens
    (("st-germain", "st germain"), "elderflower liqueur"),
    (("cointreau", "triple sec"), "orange liqueur"),
    (("grand marnier",), "orange liqueur"),
    (("luxardo maraschino", "maraschino liqueur"), "maraschino liqueur"),
    (("green chartreuse",), "green chartreuse"),
    (("yellow chartreuse",), "yellow chartreuse"),
]


def canonicalize(raw: str) -> str:
    """Raw ingredient string -> canonical fine token (a clean lowercase name)."""
    s = _strip_quantity(raw).lower().replace("\u2019", "'")
    for keys, generic in _KEYWORD_GENERIC:
        if any(k in s for k in keys):
            return generic
    return s


# ---------------------------------------------------------------------------
# 2. CATEGORY / COARSE ASSIGNMENT  (fine token -> coarse token)
# ---------------------------------------------------------------------------

# Spirit fine tokens whose coarse token is a SPIRIT FAMILY rather than the
# generic "spirit".  The whiskey subtypes collapse to coarse "whiskey";
# every other spirit is its own family.  This map is the coarse level for
# spirits and is also where the bourbon/scotch granularity fix lives.
_SPIRIT_FAMILY: dict[str, str] = {
    "bourbon": "whiskey",
    "scotch": "whiskey",
    "irish whiskey": "whiskey",
    "rye whiskey": "whiskey",
    "tennessee whiskey": "whiskey",
    "wheat whiskey": "whiskey",
    "whiskey": "whiskey",
    "gin": "gin",
    "vodka": "vodka",
    "rum": "rum",
    "tequila": "tequila",
    "mezcal": "agave",
    "cognac": "brandy",
    "brandy": "brandy",
    "armagnac": "brandy",
    "apple brandy": "brandy",
    "pisco": "brandy",
    "cachaca": "cane",
    "absinthe": "absinthe",
    "genever": "gin",
    "aquavit": "aquavit",
}
# every fine token in _SPIRIT_FAMILY is, definitionally, a spirit
_SPIRIT_FINE = set(_SPIRIT_FAMILY)

# keyword -> category, for fine tokens with no usable existing category.
# Checked top-to-bottom; first match wins, so the order encodes priority
# (a "*bitters" token is bitters even though it may contain a fruit word).
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("bitters", ("bitters",)),
    ("sweetener", ("syrup", "honey", "sugar", "grenadine", "orgeat", "agave",
                   "nectar", "molasses", "falernum", "gomme", "cordial")),
    ("fortified", ("vermouth", "sherry", "port", "madeira", "lillet",
                   "dubonnet", "quinquina", "aperitif", "americano",
                   "punt e mes", "cocchi")),
    ("wine", ("champagne", "prosecco", "cava", "wine", "sparkling",
              "cremant", "crémant", "sake")),
    ("juice", ("juice",)),
    ("dairy_egg", ("egg", "cream", "milk", "yolk", "butter", "yoghurt",
                   "yogurt")),
    ("liqueur", ("liqueur", "triple sec", "curacao", "curaçao", "schnapps",
                 "crème de", "creme de", "amaretto", "campari", "aperol",
                 "chartreuse", "bénédictine", "benedictine", "cointreau",
                 "kahlua", "st-germain", "grand marnier", "maraschino",
                 "drambuie", "frangelico", "galliano", "sambuca", "pernod",
                 "arrack", "cynar", "fernet", "amaro", "averna", "nonino",
                 "jägermeister", "jagermeister", "southern comfort",
                 "goldschlager", "limoncello")),
    ("mixer", ("soda", "tonic", "cola", "coke", "ginger ale", "ginger beer",
               "sprite", "water", "lemonade", "sour", "beer", "lager",
               "ale ", "energy drink")),
    ("fruit_herb", ("mint", "basil", "cucumber", "ginger", "cherry", "lime",
                    "lemon", "orange", "berry", "berries", "pineapple",
                    "apple", "peach", "mango", "cinnamon", "clove", "nutmeg",
                    "salt", "pepper", "chili", "chile", "leaves", "fruit",
                    "coffee", "tea", "cocoa", "chocolate", "vanilla",
                    "coconut", "cucumber")),
]

JUNK_COARSE = "[JUNK]"


def build_category_index(recipes: list[dict]) -> dict[str, str]:
    """
    For each canonical fine token, the majority of the per-ingredient
    `category` values already present in the corpus.  The original corpus
    build assigned these for ~93% of occurrences; we reuse that signal as
    the primary category source and only fall back to keyword rules for
    tokens it never covered (the new-source ingredients).
    """
    counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for r in recipes:
        for ing in r["ingredients"]:
            cat = ing.get("category")
            if cat:  # skip None / empty
                counts[canonicalize(ing["ingredient"])][cat] += 1
    return {tok: c.most_common(1)[0][0] for tok, c in counts.items()}


def coarse_token(fine: str, category_index: dict[str, str]) -> str:
    """
    The coarse token for a fine token.

      - spirit fine token  -> its spirit family  (whiskey/gin/rum/...)
      - otherwise          -> its category, from the corpus' existing
                              `category` field if present, else keyword
                              rules, else [JUNK].
    """
    if fine in _SPIRIT_FAMILY:
        return _SPIRIT_FAMILY[fine]
    cat = category_index.get(fine)
    if cat == "spirit":
        # existing field said "spirit" but we have no family for this fine
        # token -> generic spirit coarse bucket
        return "spirit"
    # the corpus' existing "other" category is a near-useless catch-all
    # (~450 tokens); it carries almost no signal, so we do NOT treat it as
    # authoritative.  Trust a SPECIFIC existing category, otherwise fall
    # through to keyword rules, and only land on "other" if those miss too.
    if cat and cat != "other":
        return cat
    for category, kws in _CATEGORY_KEYWORDS:
        if any(k in fine for k in kws):
            return category
    if cat == "other":
        return "other"   # existing field said "other" and rules agreed: keep
    return JUNK_COARSE


# ---------------------------------------------------------------------------
# 3. DRIVER
# ---------------------------------------------------------------------------

def build_vocabulary(recipes: list[dict]) -> dict:
    """Build the hierarchical vocabulary dict from the recipe list."""
    category_index = build_category_index(recipes)

    # canonical fine token -> occurrence count
    fine_counts: collections.Counter = collections.Counter()
    for r in recipes:
        for ing in r["ingredients"]:
            fine_counts[canonicalize(ing["ingredient"])] += 1

    # fine token -> coarse token
    fine_to_coarse = {
        tok: coarse_token(tok, category_index) for tok in fine_counts
    }

    # coarse vocabulary: deterministic order. [JUNK] always last so its id
    # is stable; other coarse tokens sorted by name for reproducibility.
    coarse_names = sorted(
        {c for c in fine_to_coarse.values() if c != JUNK_COARSE}
    )
    coarse_vocab = ["[PAD]", "[MASK]"] + coarse_names + [JUNK_COARSE]
    coarse_id = {name: i for i, name in enumerate(coarse_vocab)}

    # fine ingredients: most-frequent-first (id assignment order is cosmetic
    # but a stable, frequency-sorted file is easy to eyeball)
    ingredients = []
    for tok, count in fine_counts.most_common():
        coarse = fine_to_coarse[tok]
        ingredients.append(
            {
                "name": tok,
                "count": count,
                "category": category_index.get(tok),  # may be None
                "coarse": coarse,
                "coarse_id": coarse_id[coarse],
            }
        )

    return {
        "size": len(ingredients) + 2,        # +2 for [PAD],[MASK]
        "coarse_size": len(coarse_vocab),
        "coarse_vocab": coarse_vocab,
        "ingredients": ingredients,
    }


def main() -> int:
    paths = CONFIG.paths
    if not paths.recipes.exists():
        print(f"[FAIL] no corpus at {paths.recipes}")
        return 1

    print(f"loading corpus from {paths.recipes} ...")
    recipes = [json.loads(line) for line in open(paths.recipes, encoding="utf-8")]
    raw_strings = {
        ing["ingredient"] for r in recipes for ing in r["ingredients"]
    }
    print(f"  {len(recipes)} recipes, {len(raw_strings)} raw ingredient strings")

    vocab = build_vocabulary(recipes)
    n_fine = len(vocab["ingredients"])
    n_coarse = vocab["coarse_size"] - 2

    # honest coverage report
    total_occ = sum(i["count"] for i in vocab["ingredients"])
    junk = [i for i in vocab["ingredients"] if i["coarse"] == JUNK_COARSE]
    junk_occ = sum(i["count"] for i in junk)
    print(f"  canonicalized -> {n_fine} fine tokens, {n_coarse} coarse tokens")
    print(f"  [JUNK] coarse: {len(junk)} fine tokens, "
          f"{junk_occ}/{total_occ} occurrences = {junk_occ/total_occ:.2%}")

    # write vocabulary.json
    out = paths.vocabulary
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"  wrote {out}")

    # write ingredients.csv (human-readable companion)
    csv_path = paths.corpus / "ingredients.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "count", "category", "coarse", "coarse_id"])
        for i in vocab["ingredients"]:
            w.writerow([i["name"], i["count"], i["category"] or "",
                        i["coarse"], i["coarse_id"]])
    print(f"  wrote {csv_path}")

    print("\nvocabulary rebuild complete.")
    print("  next: re-run  uv run python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
