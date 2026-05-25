"""
labels.py -- derive a clean classification label for every recipe.

The corpus `category` field is 84% empty and inconsistent across
sources, so it cannot support the brief's transfer task directly.
Instead we DERIVE a label every recipe can carry: its base-spirit
family -- the highest-proportion spirit ingredient.

The transfer task built on this is deliberately NOT "read the spirit off
the ingredient list" (trivial -- the spirit is an input token). Instead
the base-spirit slot is MASKED, and the model must infer the family from
the rest of the recipe: the modifiers, citrus, sweeteners, proportions.
A pretrained encoder that genuinely learned cocktail structure should
beat a from-scratch one at that inference -- which is the transfer proof.

Only the well-populated families are kept as classes; sparse spirits and
spirit-free recipes are dropped from the labelled set.
"""

from __future__ import annotations

# the spirit families used as classification classes (each has >=450
# recipes in the corpus -- enough to train and test on)
SPIRIT_CLASSES = ["gin", "vodka", "rum", "whiskey", "tequila"]
CLASS_TO_IDX = {name: i for i, name in enumerate(SPIRIT_CLASSES)}

# the full set of spirit tokens (used to find a recipe's base spirit)
_ALL_SPIRITS = {
    "gin", "vodka", "rum", "whiskey", "tequila", "mezcal", "cognac",
    "brandy", "cachaca", "pisco", "absinthe", "applejack", "genever",
    "kirsch",
}


def base_spirit(recipe: dict) -> str | None:
    """
    The recipe's base spirit = the highest-proportion spirit ingredient.

    Returns the spirit name, or None if the recipe has no spirit. Ties
    and missing proportions fall back to the first spirit listed.
    """
    spirits = [
        (ing.get("proportion") or 0.0, idx, ing["ingredient"])
        for idx, ing in enumerate(recipe["ingredients"])
        if ing["ingredient"] in _ALL_SPIRITS
    ]
    if not spirits:
        return None
    # max by proportion; idx as a stable tiebreaker
    spirits.sort(key=lambda t: (-t[0], t[1]))
    return spirits[0][2]


def recipe_label(recipe: dict) -> int | None:
    """
    Class index for a recipe, or None if it should be excluded from the
    labelled set (no spirit, or a spirit outside the 5 classes).
    """
    sp = base_spirit(recipe)
    if sp is None:
        return None
    return CLASS_TO_IDX.get(sp)  # None if sp not one of the 5 classes


def label_recipes(recipes: list[dict]) -> list[tuple[dict, int]]:
    """
    Attach labels to recipes, keeping only those in the 5 spirit classes.

    Returns a list of (recipe, class_idx) pairs.
    """
    out = []
    for r in recipes:
        lbl = recipe_label(r)
        if lbl is not None:
            out.append((r, lbl))
    return out
