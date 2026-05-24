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
    restarts: int = 8           # independent descents; best is kept
    temp_start: float = 1.0     # softmax temperature at step 0
    temp_end: float = 0.05      # softmax temperature at the final step
    seed: int = 0
    min_count: int = 3          # only ingredients seen >= this many times
                                # are generatable -- excludes the long-tail
                                # junk tokens with barely-trained embeddings


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
) -> int:
    """
    Snap a single soft mixture to its best real ingredient, subject to
    two constraints:
      * the ingredient must be in `allowed` (the generatable pool --
        excludes long-tail junk tokens)
      * the ingredient must not be in `used` (no duplicate ingredient,
        whether the duplicate is a fixed one or an already-generated one)
    Picks the highest-logit ingredient that satisfies both.
    """
    order = torch.argsort(logits_row, descending=True).tolist()
    for idx in order:
        if idx < N_SPECIAL:
            continue
        if idx in allowed and idx not in used:
            return idx
    # fallback: allowed pool exhausted -- take best unused real ingredient
    for idx in order:
        if idx >= N_SPECIAL and idx not in used:
            return idx
    return order[0]


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

    best_energy = float("inf")
    best_ids: list[int] = []
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

        # snap each slot in turn, forbidding duplicates: a generated
        # ingredient may not repeat a fixed one or an earlier generated
        # one. `used` grows as slots are committed.
        used = set(fixed_ids)
        gen_ids: list[int] = []
        final_logits = logits.detach()
        for j in range(n_generate):
            idx = _snap_one(final_logits[j], allowed, used)
            gen_ids.append(idx)
            used.add(idx)

        # re-score honestly with the discrete energy from energy.py
        batch = _build_recipe_batch(fixed_ids + gen_ids, 0,
                                    prop_dim, max_len, device)
        true_energy = float(recipe_energy(model, batch, device=device)[0])
        per_restart.append(true_energy)

        if true_energy < best_energy:
            best_energy = true_energy
            best_ids = gen_ids

    return {
        "generated": [vocab.decode(i) for i in best_ids],
        "full_recipe": fixed_ingredients + [vocab.decode(i) for i in best_ids],
        "energy": best_energy,
        "energy_per_restart": per_restart,
    }
