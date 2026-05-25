"""
resolve.py -- fuzzy ingredient-name resolution.

The vocabulary uses specific canonical names ('whiskey', 'lime juice',
'sugar syrup'). A demo user types natural language ('bourbon', 'lime',
'simple syrup') that may not match exactly -- and an unmatched name
silently becomes a [MASK] token, quietly corrupting the result.

This resolver maps a user string to the closest canonical vocabulary
name, so 'bourbon' -> 'whiskey', 'lime' -> 'lime juice'. It uses string
similarity only (no model, no embeddings) -- cheap and predictable.

Matching strategy, in order:
  1. exact match            -> use it
  2. substring containment  -> e.g. 'lime' in 'lime juice'
  3. token-overlap + edit-distance ratio -> closest canonical name
A confidence is returned so the demo can confirm low-confidence guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from cocktail_jepa.data.vocab import Vocabulary

# Known natural-language synonyms -> canonical vocabulary names. Pure
# string similarity is semantically blind ('bourbon' shares almost no
# letters with 'whiskey'), so common domain synonyms are mapped
# explicitly. This mirrors the normalization build_corpus.py applied to
# the corpus itself. The fuzzy fallback handles anything not listed here.
SYNONYMS: dict[str, str] = {
    # whiskies -- the corpus collapses all of these to 'whiskey'
    "bourbon": "whiskey", "scotch": "whiskey", "rye": "whiskey",
    "rye whiskey": "whiskey", "bourbon whiskey": "whiskey",
    "irish whiskey": "whiskey", "scotch whisky": "whiskey",
    "whisky": "whiskey",
    # rums
    "white rum": "rum", "dark rum": "rum", "light rum": "rum",
    "spiced rum": "rum", "gold rum": "rum", "aged rum": "rum",
    # other spirits
    "blanco tequila": "tequila", "reposado tequila": "tequila",
    "anejo tequila": "tequila", "silver tequila": "tequila",
    # common modifiers
    "simple syrup": "sugar syrup", "gomme syrup": "sugar syrup",
    "sugar": "sugar syrup",
    "triple sec": "orange liqueur", "cointreau": "orange liqueur",
    "curacao": "orange liqueur", "grand marnier": "orange liqueur",
    "sweet vermouth": "vermouth", "dry vermouth": "vermouth",
    # citrus -- usually meant as the juice
    "lime": "lime juice", "lemon": "lemon juice",
    "orange": "orange juice", "grapefruit": "grapefruit juice",
    # bitters
    "angostura": "angostura bitters", "bitters": "angostura bitters",
    # bubbles
    "champagne": "sparkling wine", "prosecco": "sparkling wine",
    "soda": "soda water", "club soda": "soda water",
}


@dataclass
class Resolution:
    """Result of resolving one user-typed ingredient string."""
    query: str
    matched: str          # the canonical name chosen
    confidence: float     # 0..1
    exact: bool


def _similar(a: str, b: str) -> float:
    """Symmetric string-similarity ratio in [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


def resolve_ingredient(query: str, vocab: Vocabulary) -> Resolution:
    """
    Resolve one user-typed ingredient to the closest canonical name.
    """
    q = query.strip().lower()
    names = [n for n in vocab.id_to_token if n not in ("[PAD]", "[MASK]")]

    # 1. exact match
    if q in vocab.token_to_id:
        return Resolution(query, q, 1.0, exact=True)

    # 2. known domain synonym -- handles 'bourbon' -> 'whiskey' etc.,
    #    which pure string similarity cannot (different letters, same drink)
    if q in SYNONYMS:
        target = SYNONYMS[q]
        if target in vocab.token_to_id:
            return Resolution(query, target, 0.95, exact=False)

    # 3. substring containment -- prefer the shortest containing name
    contains = [n for n in names if q in n or n in q]
    if contains:
        best = min(contains, key=len)
        return Resolution(query, best, 0.9, exact=False)

    # 4. fuzzy: best string-similarity over the vocabulary
    scored = [(_similar(q, n), n) for n in names]
    scored.sort(reverse=True)
    score, best = scored[0]
    return Resolution(query, best, round(score, 3), exact=False)


def resolve_all(queries: list[str], vocab: Vocabulary) -> list[Resolution]:
    """Resolve a list of user ingredient strings."""
    return [resolve_ingredient(q, vocab) for q in queries]
