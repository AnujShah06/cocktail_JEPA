"""
generate.py -- constrained recipe generation by energy descent.

The third capability from the project brief: given a PARTIAL recipe
(some ingredients fixed by the user, some slots empty), complete it by
finding ingredients for the empty slots that make the whole recipe
coherent -- i.e. low energy.

Method: continuous relaxation + energy descent.
  An empty slot cannot be gradient-optimized as a discrete choice over
  ~825 ingredients. So we RELAX it: the slot is represented as a soft
  mixture -- a probability vector over the whole vocabulary -- and its
  token embedding is the weighted average of all ingredient embeddings.
  That weight vector is continuous and differentiable, so we can
  gradient-descend the recipe's energy with respect to it. A temperature
  anneal sharpens the mixtures toward one-hot over the descent. Finally
  each mixture is SNAPPED to its nearest real ingredient.

Because relaxed energy descent over a discrete space is finicky, two
safeguards from the brief are built in:
  * multi-restart   -- run several descents from different random inits,
                       keep the best.
  * discrete re-score -- the relaxed energy is only an approximation; the
                       honest score is the energy of the final REAL
                       (discrete) recipe, recomputed with energy.py.

This module does NOT train or modify the model. The JEPA is frozen; we
only descend an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.vocab import N_SPECIAL, PAD_ID, Vocabulary
from cocktail_jepa.energy.energy import recipe_energy
from cocktail_jepa.model.jepa import CocktailJEPA


@dataclass
class GenConfig:
    """Settings for the energy-descent generator."""
    steps: int = 200            # gradient steps per restart
    lr: float = 0.1             # step size on the mixture logits
    restarts: int = 8           # independent descents
    temp_start: float = 1.0     # softmax temperature at step 0
    temp_end: float = 0.05      # softmax temperature at the final step
    seed: int = 0
    min_count: int = 10         # only ingredients seen >= this many times
                                # are generatable.  The long tail (count
                                # < ~10) has under-trained embeddings and
                                # is where junk like 'chocolate ice-cream'
                                # (count 4) lives -- excluding it from
                                # GENERATION is the honest fix: do not
                                # invent from tokens the model barely saw.
                                # Scoring still accepts the full vocab.

    # --- sampling -------------------------------------------------------
    # Pure energy descent always collapses to the single lowest-energy
    # point -- the same bland, most-predictable combination every time.
    # With sample=True the generator instead SAMPLES from the low-energy
    # region: each slot is drawn from its top-k highest-logit ingredients
    # (not argmax'd), and the restart returned is sampled with probability
    # weighted toward low energy.  This trades a little optimality for
    # variety and avoids always surfacing the blandest optimum.
    # sample=False reproduces the old deterministic argmin behaviour.
    sample: bool = True
    top_k: int = 5              # per-slot: sample among the k best tokens
    sample_temp: float = 0.7    # softmax temperature for the sampling draws

    # --- cocktail grammar ----------------------------------------------
    # Sampling gives variety but not structure -- the low-energy region
    # still contains recipes with no base spirit.  With grammar=True the
    # snap step enforces a structural constraint derived from the coarse
    # ingredient taxonomy (fix #4): the finished recipe must contain a
    # base spirit, and not more than `max_base_spirits` of them.  The
    # model still freely chooses WHICH spirit and WHICH modifiers.
    #
    # (An earlier draft also capped "loose" categories to suppress junk
    # like 'chocolate ice-cream'.  That was dropped: the real junk is
    # scattered across legitimate categories -- chocolate ice-cream is in
    # dairy_egg alongside egg white -- so a category cap caused friendly
    # fire.  The honest fix for junk is the min_count floor above, which
    # excludes under-trained long-tail tokens regardless of category.)
    grammar: bool = True
    max_base_spirits: int = 2


# coarse-category grouping for the cocktail grammar.  These are coarse
# TOKEN strings as emitted by build_corpus.py; _grammar_sets() resolves
# them to coarse ids against a given vocabulary.
_BASE_SPIRIT_COARSE = {"gin", "rum", "vodka", "whiskey", "brandy",
                       "tequila", "agave", "cane", "aquavit", "spirit"}


def _grammar_sets(vocab: Vocabulary) -> set[int]:
    """Resolve the grammar's base-spirit coarse categories to coarse ids."""
    name_to_id = getattr(vocab, "coarse_token_to_id", {})
    return {name_to_id[n] for n in _BASE_SPIRIT_COARSE if n in name_to_id}


def generatable_ids(vocab: Vocabulary, min_count: int) -> set[int]:
    """
    Ingredient ids the generator is ALLOWED to produce.

    The corpus vocabulary has a long tail (~500 of ~825 tokens seen < 3
    times), a mix of genuinely rare ingredients and normalization
    artifacts ('one sugar', 'lime juice wedge'). Their embeddings are
    barely trained, so the generator should not invent them. A fixed
    --have ingredient may still be rare; this only constrains what is
    GENERATED.

    Falls back gracefully if the vocabulary carries no counts.
    """
    import json
    try:
        obj = json.load(open(CONFIG.paths.vocabulary, encoding="utf-8"))
        keep = set()
        for item in obj["ingredients"]:
            if item.get("count", 0) >= min_count:
                idx = vocab.token_to_id.get(item["name"])
                if idx is not None:
                    keep.add(idx)
        return keep if keep else set(range(N_SPECIAL, len(vocab)))
    except Exception:
        # no counts available -- allow every real ingredient
        return set(range(N_SPECIAL, len(vocab)))


def _build_recipe_batch(
    fixed_ids: list[int],
    n_empty: int,
    prop_dim: int,
    max_len: int,
    device: str,
) -> dict:
    """
    Assemble a single-recipe batch: the fixed ingredients followed by
    n_empty slots to be generated, padded to max_len.

    Proportions are set to a flat 1/n split (the generator focuses on
    *which ingredients*, not their ratios -- consistent with the Stage 4
    finding that the model is far more sensitive to ingredient identity
    than to proportion structure).
    """
    n_total = len(fixed_ids) + n_empty
    assert n_total <= max_len, f"recipe too long ({n_total} > {max_len})"

    ids = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    props = torch.zeros(1, max_len, prop_dim, dtype=torch.float32,
                        device=device)

    for i, fid in enumerate(fixed_ids):
        ids[0, i] = fid
    for i in range(n_total):
        pad_mask[0, i] = True
        # flat proportion encoding: all slots equal share, "known" flag on
        props[0, i, -1] = 1.0

    return {"ingredient_ids": ids, "proportions": props, "pad_mask": pad_mask,
            "n_ingredients": torch.tensor([n_total])}


@torch.no_grad()
def _snap_one(
    logits_row: torch.Tensor,       # [vocab_size]  one slot's mixture
    allowed: set[int],
    used: set[int],
    forbid: set[int] | None = None,
    require: set[int] | None = None,
    coarse_of=None,
) -> int:
    """
    Snap a single soft mixture to its best real ingredient, subject to:
      * the ingredient must be in `allowed` (the generatable pool)
      * the ingredient must not be in `used` (no duplicates)
      * (grammar) its coarse category must not be in `forbid`
      * (grammar) if `require` is given, its coarse category must be in it
    Picks the highest-logit ingredient satisfying every active constraint.
    """
    forbid = forbid or set()
    order = torch.argsort(logits_row, descending=True).tolist()

    def ok(idx: int) -> bool:
        if idx < N_SPECIAL or idx not in allowed or idx in used:
            return False
        if coarse_of is not None:
            c = coarse_of(idx)
            if c in forbid:
                return False
            if require is not None and c not in require:
                return False
        return True

    for idx in order:
        if ok(idx):
            return idx
    # relax `require` first (better to miss the forced category than fail)
    if require is not None:
        return _snap_one(logits_row, allowed, used, forbid=forbid,
                         require=None, coarse_of=coarse_of)
    # then relax `forbid`
    if forbid:
        return _snap_one(logits_row, allowed, used, forbid=set(),
                         require=None, coarse_of=coarse_of)
    # last resort: best unused real ingredient
    for idx in order:
        if idx >= N_SPECIAL and idx not in used:
            return idx
    return order[0]


def _snap_sampled(
    logits_row: torch.Tensor,       # [vocab_size]  one slot's mixture
    allowed: set[int],
    used: set[int],
    top_k: int,
    temp: float,
    gen: torch.Generator,
    forbid: set[int] | None = None,
    require: set[int] | None = None,
    coarse_of=None,
) -> int:
    """
    Sampling counterpart of _snap_one.

    Gathers the top_k highest-logit ingredients that satisfy every active
    constraint (allowed, unused, grammar forbid/require), then samples one
    with probability softmax(logits / temp).  The model's learned
    preference still dominates; the result is no longer deterministic.

    Falls back to _snap_one (same constraints) if fewer than two valid
    candidates exist.
    """
    forbid = forbid or set()
    order = torch.argsort(logits_row, descending=True).tolist()

    def ok(idx: int) -> bool:
        if idx < N_SPECIAL or idx not in allowed or idx in used:
            return False
        if coarse_of is not None:
            c = coarse_of(idx)
            if c in forbid:
                return False
            if require is not None and c not in require:
                return False
        return True

    cand = []
    for idx in order:
        if ok(idx):
            cand.append(idx)
        if len(cand) >= top_k:
            break
    if len(cand) < 2:
        return _snap_one(logits_row, allowed, used, forbid=forbid,
                         require=require, coarse_of=coarse_of)
    vals = torch.tensor([logits_row[i] for i in cand])
    probs = F.softmax(vals / max(temp, 1e-3), dim=0)
    pick = torch.multinomial(probs, 1, generator=gen).item()
    return cand[pick]


def _relaxed_energy(
    model: CocktailJEPA,
    fixed_ids: list[int],
    empty_logits: torch.Tensor,     # [n_empty, vocab_size], requires grad
    temperature: float,
    prop_dim: int,
    max_len: int,
    device: str,
) -> torch.Tensor:
    """
    Energy of a recipe whose empty slots are SOFT mixtures.

    The empty-slot token embedding is sum_v softmax(logits)_v * emb_v --
    a differentiable weighted average over the ingredient table. We build
    the token tensor directly and run the JEPA's own energy computation
    on it, so the relaxed energy is consistent with the discrete energy.
    """
    n_fixed = len(fixed_ids)
    n_empty = empty_logits.shape[0]
    n_total = n_fixed + n_empty

    # soft mixture weights over the vocabulary
    weights = F.softmax(empty_logits / temperature, dim=1)   # [n_empty, V]
    # exclude special tokens from the mixture
    weights = weights.clone()
    weights[:, :N_SPECIAL] = 0.0

    emb_table = model.tokens.ingredient_emb.weight           # [V, d]
    d_model = emb_table.shape[1]

    # build token embeddings: fixed slots from the table, empty slots as
    # the differentiable weighted average.
    tokens = torch.zeros(1, max_len, d_model, device=device)
    for i, fid in enumerate(fixed_ids):
        tokens[0, i] = emb_table[fid]
    for j in range(n_empty):
        tokens[0, n_fixed + j] = weights[j] @ emb_table      # [d]

    # proportion contribution: flat "known" encoding for the real slots
    props = torch.zeros(1, max_len, prop_dim, device=device)
    for i in range(n_total):
        props[0, i, -1] = 1.0
    tokens = tokens + model.tokens.proportion_gate * \
        model.tokens.proportion_proj(props)

    pad_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    pad_mask[0, :n_total] = True

    # energy = mean latent prediction error over all real slots, masking
    # each in turn (same definition as energy.py, on prebuilt tokens).
    target_emb = model.target_encoder(tokens, pad_mask)
    total_err = tokens.new_zeros(())
    for slot in range(n_total):
        ctx = model.tokens.apply_mask(tokens, torch.tensor([slot],
                                                           device=device))
        ctx_emb = model.context_encoder(ctx, pad_mask)
        query = ctx_emb[:, slot].unsqueeze(1)
        predicted = model.predictor(query, ctx_emb, pad_mask)[0]
        target = target_emb[0, slot]
        total_err = total_err + F.smooth_l1_loss(predicted, target)
    return total_err / n_total


def generate(
    model: CocktailJEPA,
    vocab: Vocabulary,
    fixed_ingredients: list[str],
    n_generate: int,
    cfg: GenConfig | None = None,
    max_len: int = 12,
    prop_dim: int = 13,
    device: str = "cpu",
) -> dict:
    """
    Complete a partial recipe by energy descent.

    fixed_ingredients : canonical ingredient names the user wants kept
    n_generate        : how many new ingredients to invent

    Returns a dict:
      generated     : list[str]   the chosen new ingredient names
      full_recipe   : list[str]   fixed + generated
      energy        : float       discrete energy of the completed recipe
      energy_per_restart : list[float]
    """
    cfg = cfg or GenConfig()
    model.eval().to(device)

    fixed_ids = [vocab.encode(name) for name in fixed_ingredients]
    vocab_size = len(vocab)

    # ingredients the generator may produce: the well-supported pool,
    # minus anything already fixed (no duplicates with --have inputs)
    allowed = generatable_ids(vocab, cfg.min_count)

    # resolve the cocktail-grammar base-spirit categories once
    grammar_base = _grammar_sets(vocab)

    # one RNG for all the sampling draws (snap + restart selection), so a
    # given cfg.seed is reproducible even in sampling mode
    sample_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 9973)

    # collect EVERY restart -- (energy, gen_ids) -- rather than only the
    # running best, so sampling mode can draw among them
    restarts: list[tuple[float, list[int]]] = []
    per_restart: list[float] = []

    for r in range(cfg.restarts):
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + r)
        logits = (torch.randn(n_generate, vocab_size, generator=g)
                  .to(device).requires_grad_(True))
        opt = torch.optim.Adam([logits], lr=cfg.lr)

        for step in range(cfg.steps):
            # temperature anneals start -> end over the descent
            frac = step / max(1, cfg.steps - 1)
            temp = cfg.temp_start + (cfg.temp_end - cfg.temp_start) * frac
            energy = _relaxed_energy(model, fixed_ids, logits, temp,
                                     prop_dim, max_len, device)
            opt.zero_grad()
            energy.backward()
            opt.step()

        # snap each slot in turn, forbidding duplicates.  In sampling mode
        # each slot is drawn from its top-k candidates; otherwise argmax.
        # With grammar on, the snap also enforces the base-spirit rule.
        used = set(fixed_ids)
        gen_ids: list[int] = []
        final_logits = logits.detach()

        # grammar bookkeeping: how many base-spirit ingredients are already
        # in the recipe (counting the fixed ones too)
        base_set = grammar_base
        def _coarse_of(fid: int) -> int:
            ci = getattr(vocab, "coarse_ids", None)
            return ci[fid] if ci and fid < len(ci) else -1
        n_base = sum(1 for fid in fixed_ids if _coarse_of(fid) in base_set)

        for j in range(n_generate):
            slots_left = n_generate - j
            forbid: set[int] = set()
            require_base = False
            if cfg.grammar:
                # cap base spirits
                if n_base >= cfg.max_base_spirits:
                    forbid |= base_set
                # if no base spirit yet and this is the last free slot,
                # REQUIRE one (force the category so the recipe has a base)
                require_base = (n_base == 0 and slots_left == 1)

            if cfg.sample:
                idx = _snap_sampled(final_logits[j], allowed, used,
                                    cfg.top_k, cfg.sample_temp, sample_gen,
                                    forbid=forbid, require=base_set
                                    if require_base else None,
                                    coarse_of=_coarse_of)
            else:
                idx = _snap_one(final_logits[j], allowed, used,
                                forbid=forbid, require=base_set
                                if require_base else None,
                                coarse_of=_coarse_of)
            gen_ids.append(idx)
            used.add(idx)
            if _coarse_of(idx) in base_set:
                n_base += 1

        # re-score honestly with the discrete energy from energy.py
        batch = _build_recipe_batch(fixed_ids + gen_ids, 0,
                                    prop_dim, max_len, device)
        true_energy = float(recipe_energy(model, batch, device=device)[0])
        per_restart.append(true_energy)
        restarts.append((true_energy, gen_ids))

    # choose the restart to return.
    #   deterministic: the strict energy minimum (old behaviour).
    #   sampling:      draw a restart with probability softmax(-energy/T),
    #                  so low-energy restarts are favoured but not forced
    #                  -- variety without abandoning the low-energy region.
    if cfg.sample and len(restarts) > 1:
        energies = torch.tensor([e for e, _ in restarts])
        probs = F.softmax(-energies / max(cfg.sample_temp, 1e-3), dim=0)
        pick = torch.multinomial(probs, 1, generator=sample_gen).item()
    else:
        pick = min(range(len(restarts)), key=lambda i: restarts[i][0])
    best_energy, best_ids = restarts[pick]

    return {
        "generated": [vocab.decode(i) for i in best_ids],
        "full_recipe": fixed_ingredients + [vocab.decode(i) for i in best_ids],
        "energy": best_energy,
        "energy_per_restart": per_restart,
    }
