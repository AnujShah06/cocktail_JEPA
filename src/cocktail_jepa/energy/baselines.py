"""
baselines.py -- trivial, non-learned energy scorers for the #43 table.

The project's headline claim is that a trained JEPA's latent prediction
error is a GOOD energy function -- it separates coherent cocktails from
incoherent ones.  "Good" is only meaningful against a floor.  These
trivial baselines are that floor: each scores a recipe with one obvious
surface statistic, no training, no model.  If the JEPA does not clearly
beat them, the energy claim is empty.

Each baseline is a function  recipe(dict) -> float  (higher = more
"incoherent", matching the JEPA energy convention).  They are scored with
the SAME evaluate_energy / AUROC machinery as the JEPA, so a baseline row
drops straight into the comparison table.

The four baselines, and what each is diagnostic FOR:

  random            -- uniform noise.  The literal floor; AUROC must come
                       out ~0.50.  It is here to sanity-check the harness
                       itself: if `random` is not ~0.5, something upstream
                       is wrong.

  length            -- score = ingredient count.  The insert and
                       incompatible_pair perturbations ADD an ingredient,
                       so length catches them for free, with zero
                       learning.  This is the honest adversary: it shows
                       how much of the JEPA's insert-type AUROC is just
                       "the recipe got one item longer".

  rarity            -- score = mean ingredient rarity (-log frequency,
                       from the vocabulary's per-ingredient `count`).  A
                       perturbation that injects an unusual ingredient
                       raises mean rarity.  Tests whether the JEPA beats
                       "this recipe contains rare ingredients".

  proportion_entropy-- score = NEGATIVE entropy of the proportion vector
                       (so lopsided proportions -> high score).
                       over_dilution makes one slot dominate (low
                       entropy -> high score), but scramble PERMUTES the
                       proportions and leaves their entropy unchanged.
                       So this baseline should catch over_dilution and be
                       blind to scramble -- which directly contextualizes
                       the JEPA's numbers on those two axes.

Nothing here trains or uses a checkpoint.
"""

from __future__ import annotations

import math
import random as _random

from cocktail_jepa.data.vocab import Vocabulary


def score_random(recipe: dict, rng: _random.Random) -> float:
    """Uniform random score in [0, 1). The floor; AUROC ~ 0.50."""
    return rng.random()


def score_length(recipe: dict) -> float:
    """Score = number of ingredients."""
    return float(len(recipe["ingredients"]))


def score_rarity(recipe: dict, log_rarity: dict[str, float],
                 default: float) -> float:
    """
    Score = mean ingredient rarity.

    log_rarity maps an ingredient string -> -log(frequency); `default`
    is used for any ingredient absent from the table (a perturbation can
    inject a string the rarity table never saw -- treating it as maximally
    rare is the natural choice and is what `default` should encode).
    """
    ings = recipe["ingredients"]
    if not ings:
        return default
    total = sum(log_rarity.get(i["ingredient"], default) for i in ings)
    return total / len(ings)


def score_proportion_entropy(recipe: dict) -> float:
    """
    Score = NEGATIVE Shannon entropy of the proportion vector.

    A balanced recipe has high proportion entropy -> low score.
    over_dilution makes one slot dominate -> low entropy -> HIGH score.
    A recipe whose proportions are all missing returns 0.0 (no signal).

    Negated so that, like every other scorer here, HIGHER == more
    incoherent.
    """
    props = [i.get("proportion") for i in recipe["ingredients"]]
    props = [p for p in props if p is not None and p > 0]
    if len(props) < 2:
        return 0.0
    total = sum(props)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p in props:
        q = p / total
        entropy -= q * math.log(q)
    return -entropy


# ---------------------------------------------------------------------------
# rarity table
# ---------------------------------------------------------------------------

def build_log_rarity(vocab: Vocabulary,
                      vocab_json: dict) -> tuple[dict[str, float], float]:
    """
    Build the ingredient -> -log(frequency) table from the vocabulary.

    The vocabulary.json carries a per-ingredient `count`.  Rarity is
    -log(count / total_count); a rarer ingredient scores higher.  Returns
    (table, default) where `default` is the rarity assigned to an unseen
    ingredient -- set to the rarity of a hypothetical count-1 ingredient,
    i.e. maximally rare.
    """
    items = vocab_json["ingredients"]
    total = sum(it["count"] for it in items) or 1
    table = {
        it["name"]: -math.log(max(it["count"], 1) / total)
        for it in items
    }
    default = -math.log(1 / total)  # an unseen ingredient ~ count 1
    return table, default


# ---------------------------------------------------------------------------
# scoring driver
# ---------------------------------------------------------------------------

# the baseline names, stable order
BASELINE_NAMES = ["random", "length", "rarity", "proportion_entropy"]


def score_recipes(
    recipes: list[dict],
    baseline: str,
    log_rarity: dict[str, float] | None = None,
    default_rarity: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """
    Score a list of recipes with one named trivial baseline.

    Returns a list of floats aligned with `recipes`.  `log_rarity` /
    `default_rarity` are required only for the 'rarity' baseline.
    """
    if baseline == "random":
        rng = _random.Random(seed)
        return [score_random(r, rng) for r in recipes]
    if baseline == "length":
        return [score_length(r) for r in recipes]
    if baseline == "rarity":
        if log_rarity is None:
            raise ValueError("rarity baseline needs a log_rarity table")
        return [score_rarity(r, log_rarity, default_rarity) for r in recipes]
    if baseline == "proportion_entropy":
        return [score_proportion_entropy(r) for r in recipes]
    raise ValueError(f"unknown baseline: {baseline!r} "
                     f"(known: {BASELINE_NAMES})")
