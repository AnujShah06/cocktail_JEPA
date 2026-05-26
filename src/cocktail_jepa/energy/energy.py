"""
energy.py -- the JEPA energy function.

The project's central claim: a trained JEPA *is* an energy-based model.
The energy of a recipe is the model's own inability to predict its parts
from one another -- the latent prediction error.

  low energy  = the model predicts the recipe's slots well
              = the recipe sits on the learned manifold of real cocktails
              = coherent
  high energy = the slots do not predict one another
              = off-manifold
              = incoherent

Estimator (decided): for a recipe of n ingredients, mask EACH slot in
turn, measure the predictor's latent error at that slot, and average the
n errors. This is deterministic -- no randomness -- so a recipe always
gets the same energy. It also covers every ingredient, so a corruption
anywhere in the recipe is seen.

This module does NOT train anything. It wraps an already-trained
CocktailJEPA and turns it into a scalar scorer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cocktail_jepa.model.jepa import CocktailJEPA


@torch.no_grad()
def recipe_energy(
    model: CocktailJEPA,
    batch: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Energy for each recipe in a batch: mean latent prediction error over
    all real ingredient slots.

    `batch` is what CocktailDataset's _stack produces (NO mask_index --
    this function supplies its own deterministic masks):
      ingredient_ids [B,L], proportions [B,L,P], pad_mask [B,L],
      n_ingredients [B]

    Returns: FloatTensor [B] -- one energy per recipe.
    """
    model.eval()
    ids = batch["ingredient_ids"].to(device)
    props = batch["proportions"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    n_ing = batch["n_ingredients"]
    B, L = ids.shape
    batch_idx = torch.arange(B, device=device)

    # embed the full recipe once; reused for every masked-slot pass
    tokens = model.tokens(ids, props)                       # [B, L, d]

    # target view: the FULL recipe through the target branch, once.
    # This must match how the model was TRAINED (see jepa.py forward):
    # with EMA the separate target encoder produced the target; without
    # EMA (the #43 ablation) the context encoder did.  Using the wrong
    # branch here would score the ablation model against an encoder it
    # never trained against.
    if getattr(model.cfg, "use_ema", True):
        target_emb = model.target_encoder(tokens, pad_mask)  # [B, L, d]
    else:
        target_emb = model.context_encoder(tokens, pad_mask)  # [B, L, d]

    # accumulate per-slot error; only real slots contribute
    err_sum = torch.zeros(B, device=device)
    slot_count = torch.zeros(B, device=device)

    max_n = int(n_ing.max().item())
    for slot in range(max_n):
        # which recipes actually have a real ingredient at this slot
        active = (slot < n_ing).to(device)                  # [B] bool
        if not active.any():
            break

        # context view: mask THIS slot, run the context encoder
        mask_index = torch.full((B,), slot, device=device, dtype=torch.long)
        context_tokens = model.tokens.apply_mask(tokens, mask_index)
        context_emb = model.context_encoder(context_tokens, pad_mask)

        # predictor: query = the masked slot's context embedding
        query = context_emb[batch_idx, mask_index].unsqueeze(1)
        predicted = model.predictor(query, context_emb, pad_mask)   # [B, d]
        target = target_emb[batch_idx, mask_index]                  # [B, d]

        # per-recipe smooth-L1 error at this slot (mean over the d dims)
        err = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=1)
        err_sum += err * active
        slot_count += active.float()

    return err_sum / slot_count.clamp(min=1.0)


@torch.no_grad()
def energy_over_loader(
    model: CocktailJEPA,
    loader,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute energy for every recipe delivered by a DataLoader.

    The loader must use a plain (non-masking) collate -- recipe_energy
    supplies its own deterministic masks. Returns (energies [N],
    recipe_ids [N]) aligned by index.
    """
    model.to(device)
    energies, ids = [], []
    for batch in loader:
        e = recipe_energy(model, batch, device=device)
        energies.append(e.cpu())
        ids.extend(batch["recipe_id"])
    return torch.cat(energies), ids
