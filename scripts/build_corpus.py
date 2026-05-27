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


# A dangling leading preposition is a parse artifact: a few raw strings
# arrive as "of club soda", "with brown sugar" -- the quantity strip
# removed a number but left the preposition.  Peel it.  (Canonicalization
# review fix.)
_LEADING_PREP_RE = re.compile(r"^(of|with|for)\s+", re.IGNORECASE)


def _strip_leading_prep(s: str) -> str:
    """Peel a dangling leading preposition left behind by quantity stripping."""
    prev = None
    while prev != s:
        prev = s
        s = _LEADING_PREP_RE.sub("", s).strip()
    return s


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



# ---------------------------------------------------------------------------
# CANONICALIZATION REVIEW: hand-fixes
# ---------------------------------------------------------------------------
# Each entry: input AFTER strip_quantity+lowercase+leading-prep -> the canonical
# fine token it should become.  217 hand-verdicts, every dirty raw string in
# the corpus reviewed individually.  Used as the FIRST lookup in canonicalize();
# rules fall through to the keyword/brand list when no override applies.

_HAND_FIXES: dict[str, str] = {
    ", orange carbonated soft drink": "orange soda",
    "10-12 oz coca-cola": "cola",
    "2-inch strips orange peel": "orange peel",
    "aberfeldy 12 year old single malt": "scotch",
    "about 8 drops tabasco sauce": "hot sauce",
    "add 10 oz root beer": "root beer",
    "add 250 ml orange juice": "orange juice",
    "algarrobo extract or malt extract from health food shops": "malt extract",
    "alvear festival pale cream sherry": "cream sherry",
    "amer picon or torani amer": "amer picon",
    "american fruits black currant cordial": "blackcurrant liqueur",
    "ancho chile-infused dolin rouge vermouth": "sweet vermouth",
    "ancho chile\u2013infused dolin rouge vermouth": "sweet vermouth",
    "apple schnapps grapefruit twist": "apple schnapps",
    "ardbeg ten 10 year old": "scotch",
    "around rim put 1 pinch sugar": "sugar",
    "bacardi ron superior limited edition": "rum",
    "balsamic vinegar of modena white": "balsamic vinegar",
    "bar code baked apple bitters": "bitters",
    "barley malt syrup 1 1": "malt syrup",
    "beavertown smog rocket smoked porter": "stout",
    "bepi tosolini grappa di moscato": "grappa",
    "bepi tosolini i legni rovere grappa": "grappa",
    "birch spirit birch eau-de- vie": "eau-de-vie",
    "bitter truth jerry thomas' bitters": "bitters",
    "bittercube cherry bark and vanilla bitters": "bitters",
    "brewed cold black breakfast tea": "black tea",
    "briottet fraise de bois liqueur": "strawberry liqueur",
    "brown sugar such as demerara": "brown sugar",
    "by the dutch batavia indonesian arrack": "arrack",
    "camomile tea cold infused with lavender": "camomile tea",
    "camomile tea syrup 2 1": "camomile tea syrup",
    "centerba 72 toro liqueur": "amaro",
    "chopped bittersweet or semi-sweet chocolate": "chocolate",
    "cinnamon orange tea\u2013infused sweet vermouth": "sweet vermouth",
    "clear creek eau de vie of douglas fir": "eau-de-vie",
    "coffee-infused carpano antica formula vermouth": "sweet vermouth",
    "cold brewed black tea preferably japanese": "black tea",
    "creme de peche peach liqueur": "peach liqueur",
    "cr\u00e8me yvette plus more for topping the drink": "creme yvette",
    "de kuyper creme de cafe liqueur": "coffee liqueur",
    "de kuyper pucker watermelon schnapps": "watermelon schnapps",
    "demerara dark muscovado brown sugar": "brown sugar",
    "dirty sue premium olive juice": "olive brine",
    "dolin g\u00e9n\u00e9py des alpes liqueur": "liqueur",
    "donn's mix #1": "spice mix",
    "donn's spices #2": "spice mix",
    "drambuie 15 liqueur": "drambuie",
    "dried and chopped angelica root": "angelica",
    "dried orange peel": "orange peel",
    "each lemon juice lime juice": "lime juice",
    "eager sicilian lemon juice soda": "lemon soda",
    "earl grey\u2013infused dolin blanc vermouth": "dry vermouth",
    "edmond briottet creme de peche liqueur": "peach liqueur",
    "edmond briottet liqueur de violette": "creme de violette",
    "fee brothers whiskey barrel-aged bitters": "bitters",
    "few slices red hot chili peppers": "chili pepper",
    "fill to top with soda water": "soda water",
    "for topping whipped cream": "whipped cream",
    "full sail session black lager": "lager",
    "garnish chocolate sauce": "chocolate sauce",
    "garnish mint": "mint",
    "garnish with blackberries": "blackberries",
    "garnish with blueberries": "blueberries",
    "garnish with kiwi": "kiwi",
    "garnish with lemon juice peel": "lemon peel",
    "garnish with lime": "lime peel",
    "garnish with mango": "mango",
    "garnish with mint": "mint",
    "garnish with orange peel": "orange peel",
    "garnish with orange spiral": "orange peel",
    "garnish with rosemary": "rosemary",
    "garnish with watermelon": "watermelon",
    "gentian liqueur e g suze salers etc": "gentian liqueur",
    "get 27": "mint liqueur",
    "giffard abricot du roussillon liqueur": "apricot liqueur",
    "giffard banane du bresil liqueur": "banana liqueur",
    "giffard creme de mure liqueur": "blackberry liqueur",
    "giffard creme de myrtille liqueur": "blueberry liqueur",
    "giffard creme de peche de vigne liqueur": "peach liqueur",
    "giffard creme de violette liqueur": "creme de violette",
    "giffard mirabelle de lorraine plum liqueur": "plum liqueur",
    "giffard pamplemousse rose pink grapefruit liqueur": "grapefruit liqueur",
    "giffard vanille de madagascar vanilla liqueur": "vanilla liqueur",
    "goslings black seal overproof 151": "overproof rum",
    "grapefruit twist": "grapefruit peel",
    "harveys porter 1859": "stout",
    "homemade apple cider syrup 2 1": "apple syrup",
    "homemade apple syrup 1 1": "apple syrup",
    "hot shot tropical fruit liqueur": "tropical fruit liqueur",
    "india pale ale ipa beer": "beer",
    "is cogas mirto liquore di sardegna": "liqueur",
    "jade perique la veritable liqueur de tabac": "liqueur",
    "joseph cartron pamplemousse rose liqueur": "grapefruit liqueur",
    "juice of 1 lime juice": "lime juice",
    "juice of a blood orange": "orange juice",
    "kalani ron de coco coconut liqueur": "coconut liqueur",
    "ketel one botanical peach orange blossom": "vodka",
    "ketel one peach orange blossom": "vodka",
    "knob creek single barrel reserve": "bourbon",
    "lejay creme de myrtille blueberry liqueur": "blueberry liqueur",
    "lejay liqueur de banane banana": "banana liqueur",
    "lemon and lime juice wheels": "lime peel",
    "lemon coin with a bit of pith": "lemon peel",
    "lemon juice and 3 oz": "lemon juice",
    "lemon juice cut into segments": "lemon",
    "lemon juice lemon twiste pomegranate seeds": "lemon juice",
    "lemon juice or orange sherbet": "lemon juice",
    "lemon juice peel": "lemon peel",
    "lemon juice syrup glasco citron": "lemon syrup",
    "lemon juice wedge cinnamon sugar": "lemon",
    "lemon juice wedge superfine sugar": "lemon",
    "lemon peel": "lemon peel",
    "lemon twist": "lemon peel",
    "lemon wedge superfine sugar long wide spiral of orange zest": "lemon",
    "lemon- lime juice soda sprite 7-up": "lemon-lime soda",
    "lemon-lime juice soda sprite 7- up": "lemon-lime soda",
    "lemon-lime juice soda sprite 7-up": "lemon-lime soda",
    "licor 43 original liqueur": "vanilla liqueur",
    "lime cordial sweetened lime juice": "lime cordial",
    "lime juice cut into 4 wedges": "lime",
    "lime juice cut into segments": "lime",
    "lime juice cut into small wedges": "lime",
    "lime juice in a highball glass": "lime juice",
    "lime juice lime juice wheel": "lime juice",
    "lime juice reserve 1 2 lime juice shell for garnish": "lime juice",
    "lime juice shell spent lime juice husk": "lime peel",
    "lime juice wedge": "lime",
    "lime juice wedge pink sanding sugar": "lime",
    "lime juice wedge superfine sugar": "lime",
    "lime juice wedge sweet chili powder": "chili pepper",
    "lime juice wheel": "lime peel",
    "lime juice zest peel": "lime peel",
    "lime peel": "lime peel",
    "little bit of blackcurrant squash": "blackcurrant",
    "lustau east india solera sherry": "sherry",
    "lustau los arcos amontillado sherry": "amontillado sherry",
    "manzoni rosa extra dry spumante": "sparkling wine",
    "marie brizard white cr\u00e8me de cacao": "white creme de cacao",
    "marie brizard white cr\u00e8me de menthe": "white creme de menthe",
    "marsala superiore doc secco wine": "marsala wine",
    "massenez cr\u00e8me de m\u00fbre blackberry liqueur": "blackberry liqueur",
    "massenez cr\u00e8me de p\u00eache peach liqueur": "peach liqueur",
    "merlet cr\u00e8me de fraise des bois strawberry liqueur": "strawberry liqueur",
    "mozart white chocolate vanilla cream liqueur": "chocolate liqueur",
    "mr boston creme de noyaux": "creme de noyaux",
    "nardini acqua de cedro liqueur": "lemon liqueur",
    "ocean spray classic cranberry drink": "cranberry juice",
    "of brown sugar": "brown sugar",
    "of lemon peel": "lemon peel",
    "of orange peel": "orange peel",
    "olive brine from jarred olive": "olive brine",
    "or lime lemon": "lemon",
    "or tinned apricot": "apricot",
    "orange fruit cut into segments": "orange",
    "orange half-wheel": "orange peel",
    "orange peel": "orange peel",
    "orange twist": "orange peel",
    "orange wheel": "orange peel",
    "otima 10-year tawny port": "port",
    "ouzo 12": "ouzo",
    "peach ripe - skinned diced": "peach",
    "peach ripe - skinned diced chopped": "peach",
    "peppercorn syrup 1 1 homemade": "peppercorn syrup",
    "pint sweet or dry cider": "cider",
    "pomegranate molasses available at middle eastern grocers": "pomegranate molasses",
    "prune syrup from tinned fruit": "prune syrup",
    "q ng xi ng f n xi ng light fragrance baijiu": "baijiu",
    "racines de suze gentian liqueur": "gentian liqueur",
    "rangpur": "rangpur lime",
    "rangpur citrus limonia limao-capeta limao-cravo fruit": "rangpur lime",
    "red jalapeno fresno chili 10 000 shu deseeded chopped": "chili pepper",
    "red jalapeno fresno chili 10 000 shu deseeded chopped chopped": "chili pepper",
    "red jalapeno fresno chili 10 000 shu deseeded chopped fine sliced": "chili pepper",
    "rothman & winter apricot liqueur": "apricot liqueur",
    "rothman & winter cherry liqueur": "cherry liqueur",
    "rothman & winter cr\u00e8me de violette": "creme de violette",
    "rothman & winter pear liqueur": "pear liqueur",
    "rothman winter creme de violette liqueur": "creme de violette",
    "rothman winter orchard pear liqueur": "pear liqueur",
    "ruby red grapefruit chopped wedges": "grapefruit",
    "rutte zn 12 oude graan jenever": "genever",
    "s mrus chai cream liqueur": "cream liqueur",
    "small fresh kaffir lime leaf": "kaffir lime leaf",
    "small ripe anjou pear slices": "pear",
    "solerno delicato blood orange liqueur": "orange liqueur",
    "solo coffee cold brew concentrate": "cold brew coffee",
    "southern usa- style liqueur 40 e g southern comfort": "southern comfort",
    "southern usa-style liqueur 40 e g southern comfort": "southern comfort",
    "spicy sugar and salt rim": "sugar",
    "st elizabeth allspice dram liqueur": "allspice dram",
    "st george spiced pear liqueur": "pear liqueur",
    "strips lemon peel": "lemon peel",
    "sugar ground in mortar and pestle": "sugar",
    "suntory yamazaki 12 year old": "japanese whiskey",
    "suze gentian liqueur infused with coffee beans": "gentian liqueur",
    "tangerine mandarin clementine juice juice": "tangerine juice",
    "tarragon and agave nectar gastrique": "agave nectar",
    "tea - strong white tea cold": "white tea",
    "tea syrup breakfast tea 2 1": "tea syrup",
    "tempus fugit creme de noyaux liqueur": "creme de noyaux",
    "tempus fugit gran classico bitter": "amaro",
    "tempus fugit gran classico bitter liqueur": "amaro",
    "the bitter truth creme de violette": "creme de violette",
    "the bitter truth wood drops and dashes": "bitters",
    "thin slices red chili pepper": "chili pepper",
    "thomas henry bitter lemon juice": "bitter lemon",
    "thomas henry pink grapefruit soda": "grapefruit soda",
    "three cents pink grapefruit soda": "grapefruit soda",
    "top it up with tonic water": "tonic water",
    "toschi nocello walnut liqueur de noix": "walnut liqueur",
    "vanilla liqueur and 3 oz": "vanilla liqueur",
    "vanilla syrup preferably b a reynold's": "vanilla syrup",
    "vieille de prune eau-de- vie": "eau-de-vie",
    "wide spiral of lemon zest": "lemon peel",
    "williams & humbert dry sack medium sherry": "amontillado sherry",
    "zucca rabarbaro marca depositata specialita dal 1845": "amaro",
}

def canonicalize(raw: str) -> str:
    """Raw ingredient string -> canonical fine token (a clean lowercase name)."""
    s = _strip_quantity(raw).lower().replace("\u2019", "'")
    s = _strip_leading_prep(s)
    # explicit hand-fix override (canonicalization review): try this first,
    # so per-string judgments win over the keyword/brand rules below.
    if s in _HAND_FIXES:
        return _HAND_FIXES[s]
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
