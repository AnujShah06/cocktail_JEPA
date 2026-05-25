"""
perturb.py -- build the held-out perturbation set.

Stage 4 evaluates the energy function by asking: can it separate real
recipes from incoherent ones? That requires a set of KNOWN-bad recipes.
None exist as natural data, so we synthesize them by corrupting real
TEST recipes -- and only test recipes, so nothing here ever touches
training or model selection.

PERTURBATION TYPES (Phase-1 fix #8 expands 3 -> 7)
--------------------------------------------------
The original three are SINGLE-SLOT corruptions of an otherwise-intact
recipe -- they probe "is one slot wrong?".  jepa-04 caught ingredient
swaps easily (substitute/insert AUROC ~0.72-0.76) but was near-blind to
proportion errors (scramble ~0.57).  The expanded set adds harder
single-slot negatives that target specific failure modes, plus one
GLOBAL-coherence negative (recombine) that no single-slot type can probe.

  Original (single-slot):
    substitute  -- swap one ingredient for a random unrelated one
    scramble    -- shuffle the proportions across slots (right ingredients,
                   wrong balance)
    insert      -- add a random extra ingredient that does not belong

  New, single-slot, harder / more diagnostic:
    category_violation -- swap an ingredient for one from a DIFFERENT
                   coarse category (a juice becomes a spirit).  A more
                   plausible-looking error than a fully random swap;
                   probes coarse-level coherence.  Uses the hierarchical
                   vocabulary (#4).
    over_dilution -- inflate a low-impact slot's proportion (water / juice
                   / mixer) so the drink is mostly diluent.  Every
                   ingredient is right; only the balance is broken.
                   Directly targets the proportion-sensitivity weakness.
    incompatible_pair -- inject an ingredient that the corpus genuinely
                   AVOIDS pairing with something already present (mined
                   by co-occurrence lift, not raw zero-counts).  The
                   hardest single-slot negative; probes learned affinity.

  New, GLOBAL coherence:
    recombine   -- splice the front of one real recipe onto the back of
                   another (different base spirit, to limit accidental
                   coherence).  Every ingredient and every proportion is
                   real and in-distribution; the recipe AS A WHOLE is
                   not.  No single-slot type tests this.  Honestly: these
                   are non-recipes assembled from real recipe fragments,
                   NOT "real bad cocktails".

Each perturbed recipe is tagged with its type and its source recipe id,
so Stage 4 reports discrimination per perturbation type.  evaluate.py
iterates the type tags dynamically, so the four new types flow into the
per-type AUROC report with no change there.
"""

from __future__ import annotations

import collections
import copy
import json
import random
from itertools import combinations
from pathlib import Path

from cocktail_jepa.data.vocab import Vocabulary

# the seven perturbation type tags, in a stable order
PERTURBATION_TYPES = [
    "substitute",
    "scramble",
    "insert",
    "category_violation",
    "over_dilution",
    "incompatible_pair",
    "recombine",
]

# coarse categories treated as low-impact "diluent" for over_dilution
_DILUENT_COARSE = {"mixer", "juice"}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _all_ingredient_names(recipes: list[dict]) -> list[str]:
    """Pool of canonical ingredient names seen across the recipes."""
    names: set[str] = set()
    for r in recipes:
        for i in r["ingredients"]:
            names.add(i["ingredient"])
    return sorted(names)


def _coarse_lookup(vocab: Vocabulary | None) -> dict[str, str]:
    """
    Map ingredient name -> coarse-token name, via the hierarchical
    vocabulary.  Returns {} if no vocab is supplied (then any perturbation
    that needs coarse info degrades to a no-op for that recipe).
    """
    if vocab is None:
        return {}
    out: dict[str, str] = {}
    for name, fine_id in vocab.token_to_id.items():
        coarse_id = vocab.coarse_of(fine_id)
        out[name] = vocab.coarse_id_to_token[coarse_id]
    return out


# ---------------------------------------------------------------------------
# original three perturbations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# new single-slot perturbations
# ---------------------------------------------------------------------------

def perturb_category_violation(
    recipe: dict,
    by_coarse: dict[str, list[str]],
    coarse_of: dict[str, str],
    rng: random.Random,
) -> dict:
    """
    Swap one ingredient for one from a DIFFERENT coarse category.

    Picks a slot whose coarse category is known, then replaces its
    ingredient with one drawn from a different coarse bucket -- e.g. a
    citrus juice becomes a spirit.  A more plausible-looking error than a
    fully random substitution, and it specifically probes whether the
    energy registers coarse-category coherence.

    Falls back to a no-op if coarse info is unavailable or the recipe has
    no slot with a usable coarse category.
    """
    out = copy.deepcopy(recipe)
    if not by_coarse:
        return out
    # slots whose ingredient has a real (non-JUNK) coarse category
    typed = [
        idx for idx, ing in enumerate(out["ingredients"])
        if coarse_of.get(ing["ingredient"], "[JUNK]") not in ("[JUNK]",)
    ]
    if not typed:
        return out
    slot = rng.choice(typed)
    own_coarse = coarse_of[out["ingredients"][slot]["ingredient"]]
    other_coarses = [c for c in by_coarse if c not in (own_coarse, "[JUNK]")]
    if not other_coarses:
        return out
    present = {i["ingredient"] for i in out["ingredients"]}
    rng.shuffle(other_coarses)
    for c in other_coarses:
        choices = [n for n in by_coarse[c] if n not in present]
        if choices:
            out["ingredients"][slot]["ingredient"] = rng.choice(choices)
            out["ingredients"][slot]["category"] = "perturbed"
            return out
    return out


def perturb_over_dilution(recipe: dict, coarse_of: dict[str, str],
                          rng: random.Random) -> dict:
    """
    Inflate a diluent slot's proportion so the drink is mostly diluent.

    Picks a slot whose coarse category is a diluent (water / mixer /
    juice), sets its proportion very high (0.80-0.92), and renormalizes
    the remaining slots to share the rest.  Every ingredient is correct;
    only the balance is broken -- this targets the proportion-sensitivity
    weakness that scramble alone could not move.

    Falls back to a no-op if there is no diluent slot or only one slot.
    """
    out = copy.deepcopy(recipe)
    ings = out["ingredients"]
    if len(ings) < 2:
        return out
    diluent_slots = [
        idx for idx, ing in enumerate(ings)
        if coarse_of.get(ing["ingredient"]) in _DILUENT_COARSE
        or "water" in ing["ingredient"].lower()
    ]
    if not diluent_slots:
        return out
    slot = rng.choice(diluent_slots)
    big = rng.uniform(0.80, 0.92)
    rest = 1.0 - big
    others = [i for i in range(len(ings)) if i != slot]
    share = rest / len(others)
    for i in range(len(ings)):
        ings[i]["proportion"] = round(big if i == slot else share, 4)
    ings[slot]["category"] = "perturbed"
    return out


def perturb_incompatible_pair(
    recipe: dict,
    avoided: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    """
    Inject an ingredient the corpus genuinely AVOIDS pairing with one
    already present.

    `avoided` maps an ingredient -> ingredients it almost never co-occurs
    with (mined by co-occurrence lift, so both partners are common enough
    that we WOULD expect them together -- a real avoidance signal, not a
    rare-token artifact).  The hardest single-slot negative: the injected
    ingredient is itself a perfectly normal cocktail ingredient; only the
    PAIRING is wrong.

    Falls back to a no-op if no present ingredient has a mined partner.
    """
    out = copy.deepcopy(recipe)
    present = {i["ingredient"] for i in out["ingredients"]}
    anchors = [n for n in present if n in avoided]
    if not anchors:
        return out
    rng.shuffle(anchors)
    for anchor in anchors:
        partners = [p for p in avoided[anchor] if p not in present]
        if partners:
            out["ingredients"].append({
                "ingredient": rng.choice(partners),
                "qty_oz": None,
                "category": "perturbed",
                "proportion": None,
            })
            out["n_ingredients"] = len(out["ingredients"])
            return out
    return out


# ---------------------------------------------------------------------------
# new global-coherence perturbation
# ---------------------------------------------------------------------------

def perturb_recombine(
    recipe: dict,
    donors: list[dict],
    rng: random.Random,
) -> dict:
    """
    Splice the front of `recipe` onto the back of a donor recipe.

    Takes the first half of this recipe's ingredients and the second half
    of a donor's, concatenates them, and renormalizes proportions so they
    sum to 1.  Every ingredient and proportion comes from a real recipe,
    so nothing is locally out of distribution -- but the recipe as a
    WHOLE is a non-recipe.  This is the only perturbation that probes
    global coherence rather than a single corrupted slot.

    `donors` should be recipes with a DIFFERENT base spirit than `recipe`
    (the caller arranges this) to limit accidental coherence.  Falls back
    to a no-op if no usable donor is available.
    """
    if not donors:
        return copy.deepcopy(recipe)
    donor = rng.choice(donors)
    a = recipe["ingredients"]
    b = donor["ingredients"]
    cut_a = max(1, len(a) // 2)
    cut_b = max(1, len(b) // 2)
    front = copy.deepcopy(a[:cut_a])
    back = copy.deepcopy(b[cut_b:])
    merged = front + back
    if len(merged) < 2:
        merged = copy.deepcopy(a[:1]) + copy.deepcopy(b[:1])

    # renormalize the proportions that exist; leave None as None
    known = [i for i in merged if i.get("proportion") is not None]
    total = sum(i["proportion"] for i in known)
    if total > 0:
        for i in known:
            i["proportion"] = round(i["proportion"] / total, 4)

    out = {
        "recipe_id": recipe.get("recipe_id", ""),
        "name": f"{recipe.get('name', '?')} x {donor.get('name', '?')}",
        "source": "recombine",
        "category": None,
        "n_ingredients": len(merged),
        "ingredients": merged,
    }
    return out


# ---------------------------------------------------------------------------
# mining: build the lift-based avoided-pair table
# ---------------------------------------------------------------------------

def mine_avoided_pairs(
    recipes: list[dict],
    min_count: int = 40,
    min_expected: float = 5.0,
    max_lift: float = 0.15,
) -> dict[str, list[str]]:
    """
    Mine ingredient pairs the corpus genuinely AVOIDS.

    For every pair of ingredients each seen >= min_count times, compute
    lift = observed_cooccurrences / expected_under_independence.  A pair
    with expected >= min_expected (we'd really expect them together) but
    lift < max_lift (they essentially never actually co-occur) is a
    genuine avoidance -- not a rare-token artifact.

    Returned as {ingredient: [avoided partners]} for fast lookup.
    """
    n_recipes = len(recipes)
    fine_count: collections.Counter = collections.Counter()
    for r in recipes:
        for i in r["ingredients"]:
            fine_count[i["ingredient"]] += 1
    common = {n for n, c in fine_count.items() if c >= min_count}

    pair: collections.Counter = collections.Counter()
    for r in recipes:
        ings = sorted({i["ingredient"] for i in r["ingredients"]} & common)
        for a, b in combinations(ings, 2):
            pair[(a, b)] += 1

    avoided: dict[str, list[str]] = collections.defaultdict(list)
    for a, b in combinations(sorted(common), 2):
        expected = fine_count[a] * fine_count[b] / max(1, n_recipes)
        if expected >= min_expected:
            lift = pair[(a, b)] / expected
            if lift < max_lift:
                avoided[a].append(b)
                avoided[b].append(a)
    return dict(avoided)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _ingredient_signature(recipe: dict) -> list:
    """(ingredient, proportion) per slot -- identity of a recipe's content."""
    return [
        (i["ingredient"], i.get("proportion")) for i in recipe["ingredients"]
    ]


def make_perturbation_set(
    test_recipes: list[dict],
    seed: int = 42,
    vocab: Vocabulary | None = None,
    mining_recipes: list[dict] | None = None,
) -> list[dict]:
    """
    Build the perturbation set from test recipes.

    Each test recipe produces AT MOST one perturbed copy per perturbation
    type (seven types).  A perturbation that could not change the recipe
    (e.g. over_dilution on a recipe with no diluent slot, scramble on
    all-equal proportions) is a NO-OP -- it would otherwise emit an
    unmodified real recipe carrying a "perturbed" label, which is pure
    noise in that type's AUROC.  Such no-ops are DROPPED, so the set may
    have slightly fewer than 7 x N entries.  The per-type count is
    therefore reported by prepare_data.py.

    Returns a flat list; each item carries:
      perturbation : one of PERTURBATION_TYPES
      source_id    : recipe_id of the real recipe it was derived from

    vocab            : the hierarchical Vocabulary -- enables the
                       category_violation and over_dilution types.  If
                       None, those two degrade to no-ops (so the function
                       still runs, just with weaker negatives).
    mining_recipes   : the recipe pool to mine avoided pairs from.  Pass
                       the FULL corpus here so the co-occurrence statistics
                       are well-estimated; defaults to test_recipes if not
                       given (weaker, but never leaks labels -- the mining
                       uses no model and no held-out signal).
    """
    rng = random.Random(seed)
    pool = _all_ingredient_names(test_recipes)
    coarse_of = _coarse_lookup(vocab)

    # ingredient names grouped by coarse category, for category_violation
    by_coarse: dict[str, list[str]] = collections.defaultdict(list)
    for name in pool:
        c = coarse_of.get(name)
        if c:
            by_coarse[c].append(name)

    # mined avoided-pair table, for incompatible_pair
    avoided = mine_avoided_pairs(mining_recipes or test_recipes)

    # base-spirit index, for recombine donor selection
    try:
        from cocktail_jepa.transfer.labels import base_spirit
        spirit_of = {id(r): base_spirit(r) for r in test_recipes}
    except Exception:
        spirit_of = {}

    out: list[dict] = []
    for recipe in test_recipes:
        # recombine donors: test recipes with a different base spirit
        own_spirit = spirit_of.get(id(recipe))
        donors = [
            r for r in test_recipes
            if r is not recipe
            and spirit_of.get(id(r)) != own_spirit
        ] if spirit_of else [r for r in test_recipes if r is not recipe]

        builders = [
            ("substitute",
             lambda r: perturb_substitute(r, pool, rng)),
            ("scramble",
             lambda r: perturb_scramble(r, rng)),
            ("insert",
             lambda r: perturb_insert(r, pool, rng)),
            ("category_violation",
             lambda r: perturb_category_violation(r, by_coarse, coarse_of, rng)),
            ("over_dilution",
             lambda r: perturb_over_dilution(r, coarse_of, rng)),
            ("incompatible_pair",
             lambda r: perturb_incompatible_pair(r, avoided, rng)),
            ("recombine",
             lambda r: perturb_recombine(r, donors, rng)),
        ]
        original_sig = _ingredient_signature(recipe)
        for kind, fn in builders:
            p = fn(recipe)
            # drop no-ops: a "perturbed" recipe identical to its source is
            # not a negative, it is noise.  recombine is exempt -- its
            # output is a new spliced recipe, never compared slot-for-slot
            # to the source (and its name/source fields already differ).
            if kind != "recombine" and _ingredient_signature(p) == original_sig:
                continue
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
