"""
pmi_baseline.py -- a competent, simple, LEARNED coherence energy.

This fills the rung the #43 baseline ladder skips.  The trivial baselines
(random / length / rarity / proportion_entropy) have no parameters and no
training; the deep baselines (JEPA / MAE / contrastive) are 2M-parameter
Transformers.  Between them sits the question a skeptic actually asks:

    "Does a 50-line learned model -- no deep network, no GPU -- already
     capture most of the coherence signal?  If so, what are the 2.1M
     parameters buying?"

The PMI energy answers it.  It learns one thing from the TRAINING split:
how often each pair of ingredients co-occurs, relative to chance.  That is
pointwise mutual information,

    PMI(a, b) = log( P(a, b) / ( P(a) * P(b) ) ),

high when two ingredients appear together more than independence predicts
(gin & tonic), low/negative when they avoid each other (gin & milk).  A
coherent recipe is one whose ingredient pairs are mutually expected, so its
COHERENCE is the mean pairwise PMI over all pairs in the recipe.

The harness convention is HIGHER == more incoherent (matching the JEPA
energy), so the ENERGY is the negative mean pairwise PMI.

Design decisions, each defensible in a deep-dive:

  * Fit on TRAIN only.  The test recipes and their perturbations are never
    seen while estimating co-occurrence -- otherwise the baseline would
    peek at the very recipes it is scored on.  This mirrors how the JEPA
    is trained on train and evaluated on held-out test.

  * Pairs, not singletons.  A unigram model (how common is each ingredient)
    is already covered by the `rarity` trivial baseline.  PMI is the
    cheapest model that captures INTERACTION -- which ingredients belong
    TOGETHER -- which is what "coherence" means.  This is deliberately the
    simplest learned model that is not redundant with an existing row.

  * add-k smoothing.  Raw PMI is undefined for a never-co-occurring pair
    (log 0).  A perturbation that injects an out-of-distribution pair must
    get a finite, low score, not -inf.  add-k (k=0.5) on the pair counts
    gives unseen pairs a strongly-negative-but-finite PMI -- exactly the
    "these don't go together" signal we want.

  * Unknown ingredients.  A perturbation can inject an ingredient string
    the train split never saw (it has no unigram count).  Such a token
    co-occurs with nothing, so every pair through it hits the add-k floor
    -- which correctly reads as incoherent.  No special-casing needed.

Nothing here uses torch for fitting; it is plain dict arithmetic.  The
scorer returns one float per recipe, so it drops into the SAME
evaluate_energy / AUROC machinery as every other row in the table.
"""

from __future__ import annotations

import math
from itertools import combinations


class PMIEnergy:
    """
    A pairwise-PMI coherence energy fitted on a training corpus.

    Usage:
        pmi = PMIEnergy.fit(train_recipes, smoothing=0.5)
        scores = pmi.score_recipes(test_recipes)   # higher == incoherent
    """

    def __init__(
        self,
        pair_pmi: dict[frozenset[str], float],
        unigram: dict[str, int],
        n_recipes: int,
        p_ab_floor: float,
        smoothing: float,
    ):
        # PMI for every pair seen in training (symmetric, keyed by a
        # 2-element frozenset so {a,b} == {b,a}).
        self._pair_pmi = pair_pmi
        # per-ingredient document frequency (how many recipes contain it).
        self._unigram = unigram
        self._n_recipes = n_recipes
        # P(a,b) for an unseen pair (smoothed-zero pair mass).  The per-pair
        # floor PMI = log(p_ab_floor / (P(a) * P(b))) is computed at scoring
        # time using the ACTUAL unigrams of the pair -- see _pair_score.
        self._p_ab_floor = p_ab_floor
        self._smoothing = smoothing
        # fallback unigram probability for an ingredient absent from train.
        # Treat it as a hypothetical "appeared in one recipe": maximally
        # rare but finite, so the PMI of an unseen-ingredient pair is the
        # log of a tiny numerator over a not-tinier denominator (negative).
        self._p_unigram_unseen = 1.0 / (n_recipes + 1)

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        recipes: list[dict],
        smoothing: float = 0.5,
    ) -> "PMIEnergy":
        """
        Estimate co-occurrence PMI from a list of recipes (the TRAIN split).

        Probabilities are estimated as document frequencies: P(a) is the
        fraction of recipes containing ingredient a; P(a,b) the fraction
        containing both.  add-k smoothing (k=`smoothing`) is applied to the
        PAIR counts so a pair that never co-occurs still gets a finite,
        low PMI rather than log(0).
        """
        n = len(recipes)
        if n == 0:
            raise ValueError("cannot fit PMIEnergy on an empty corpus")

        unigram: dict[str, int] = {}
        pair_count: dict[frozenset[str], int] = {}

        for r in recipes:
            # de-duplicate ingredients within a recipe: co-occurrence is a
            # set relation, a doubled ingredient must not double-count.
            ings = {ing["ingredient"] for ing in r["ingredients"]}
            for a in ings:
                unigram[a] = unigram.get(a, 0) + 1
            for a, b in combinations(sorted(ings), 2):
                key = frozenset((a, b))
                pair_count[key] = pair_count.get(key, 0) + 1

        k = smoothing
        # The probability model:  every probability is per-RECIPE, so the
        # PMI ratio  P(a,b) / (P(a)*P(b))  is well-defined.
        #   P(a)   = recipes containing a    / n_recipes
        #   P(b)   = recipes containing b    / n_recipes
        #   P(a,b) = recipes containing both / n_recipes
        # Smoothing: add k "phantom" recipes per pair, so an unseen pair has
        # k/(n + k*n_possible) probability rather than zero.
        vocab = list(unigram)
        n_possible_pairs = len(vocab) * (len(vocab) - 1) / 2 or 1.0
        # smoothed effective number of "recipes" in the joint distribution:
        # real recipe count + k phantom recipes per possible pair.
        n_eff = n + k * n_possible_pairs

        def p_unigram(a: str) -> float:
            return unigram.get(a, 0) / n

        pair_pmi: dict[frozenset[str], float] = {}
        for key, c in pair_count.items():
            a, b = tuple(key) if len(key) == 2 else (next(iter(key)),) * 2
            p_ab = (c + k) / n_eff
            pa, pb = p_unigram(a), p_unigram(b)
            if pa <= 0 or pb <= 0:
                continue
            pair_pmi[key] = math.log(p_ab / (pa * pb))

        # P(a,b) for an unseen pair: smoothed-zero count over n_eff.
        # The per-pair floor PMI = log(p_ab_floor / (P(a) * P(b))) is computed
        # at scoring time from the ACTUAL unigrams of a and b -- see
        # _pair_score.  This is correct PMI semantics: two common ingredients
        # that never co-occur (gin + milk) get a STRONGLY NEGATIVE PMI, while
        # two rare ingredients that never co-occur get a less-strongly-
        # negative PMI (with weak unigram evidence the incoherence claim is
        # weaker -- this is the right behavior).
        p_ab_floor = k / n_eff

        return cls(
            pair_pmi=pair_pmi,
            unigram=unigram,
            n_recipes=n,
            p_ab_floor=p_ab_floor,
            smoothing=smoothing,
        )

    # -- scoring -----------------------------------------------------------

    def _p_unigram(self, a: str) -> float:
        """P(a) = document frequency / corpus size, with a small-positive
        fallback for ingredients absent from training so PMI stays finite."""
        c = self._unigram.get(a, 0)
        if c == 0:
            return self._p_unigram_unseen
        return c / self._n_recipes

    def _pair_score(self, a: str, b: str) -> float:
        """
        PMI of a single pair.

        For pairs SEEN in training: the stored, observed PMI -- positive for
        pairs that co-occur more than independence predicts, negative for
        pairs that co-occur less than independence predicts.

        For pairs UNSEEN in training: compute the floor PMI from the actual
        unigrams of a and b, log(p_ab_floor / (P(a)*P(b))).  Two common
        ingredients that never co-occur (gin + milk) hit a STRONGLY NEGATIVE
        PMI here -- which is the semantically correct "these don't go
        together" signal.  Two rare ingredients that never co-occur get a
        less-strongly-negative PMI, which is also right: with weak unigram
        evidence we cannot make a strong incoherence claim.
        """
        key = frozenset((a, b))
        if key in self._pair_pmi:
            return self._pair_pmi[key]
        pa = self._p_unigram(a)
        pb = self._p_unigram(b)
        return math.log(self._p_ab_floor / (pa * pb))

    def coherence(self, recipe: dict) -> float:
        """
        Mean pairwise PMI over the recipe's ingredient pairs.

        HIGH coherence == ingredients mutually expected to co-occur.
        A 0- or 1-ingredient recipe has no pairs; it returns the floor,
        treating a degenerate recipe as minimally coherent.
        """
        ings = sorted({ing["ingredient"] for ing in recipe["ingredients"]})
        if len(ings) < 2:
            # Degenerate: no pairs to score.  Return 0 so the energy is 0 --
            # this is rare in practice (the dataset filters to >=2 ingredients
            # anyway) and a neutral score is the honest non-claim.
            return 0.0
        pmis = [self._pair_score(a, b) for a, b in combinations(ings, 2)]
        return sum(pmis) / len(pmis)

    def energy(self, recipe: dict) -> float:
        """
        Coherence ENERGY: negative mean pairwise PMI.

        Higher == more incoherent, matching the JEPA energy convention so
        this drops straight into evaluate_energy as another row.
        """
        return -self.coherence(recipe)

    def score_recipes(self, recipes: list[dict]) -> list[float]:
        """Energy for each recipe, aligned with the input list."""
        return [self.energy(r) for r in recipes]
