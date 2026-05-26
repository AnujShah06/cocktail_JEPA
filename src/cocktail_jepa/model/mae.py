"""
mae.py -- a Masked-AutoEncoder baseline, the MAE row of the #43 table.

WHY THIS EXISTS
---------------
The project's headline claim is that a JEPA's LATENT prediction error is a
good energy function.  A reviewer's first question is: is it the JEPA
*objective* that matters, or would ANY masked self-supervised encoder
give an equally good energy?  The MAE answers that.  It is built to be
identical to the JEPA in every way EXCEPT the objective:

  same TokenEncoder (hierarchical coarse+fine ingredient + proportion),
  same SetEncoder capacity (the encoder backbone is the same class/size),
  same training budget.

The ONE difference -- and the whole point of the comparison:

  JEPA : mask a slot, predict its LATENT (an embedding), measure latent
         prediction error.  No decoding back to ingredients.
  MAE  : mask a slot, RECONSTRUCT it -- predict the ingredient-ID
         distribution and the proportion scalar -- measure reconstruction
         error in INPUT space.

So the MAE drops everything specific to the JEPA's joint-embedding
objective (the EMA target encoder, the predictor, the SIGReg regularizer
-- a masked autoencoder needs none of them; reconstruction cannot
collapse, because a constant output cannot reconstruct varied inputs) and
adds a reconstruction decoder head.

ENERGY
------
The MAE's energy of a recipe is the natural analog of the JEPA's: mask
each slot in turn, measure how badly the decoder reconstructs it, average
over slots.  High reconstruction error = the recipe's slots do not
predict one another = off-manifold = incoherent.  `recipe_energy_mae`
implements this and is the MAE counterpart of energy.recipe_energy, so
the #43 table can score JEPA and MAE rows the same way.

This is a BASELINE: kept deliberately minimal and matched to the JEPA, so
the comparison is fair.  It is not tuned to win.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocktail_jepa.model.encoder import SetEncoder
from cocktail_jepa.model.tokens import TokenEncoder


@dataclass
class MAEConfig:
    """Architecture for the MAE baseline -- mirrors JEPAConfig's encoder."""
    vocab_size: int
    prop_dim: int
    d_model: int = 192
    enc_layers: int = 3
    enc_heads: int = 6
    dropout: float = 0.1
    # weight on the proportion-reconstruction term, relative to the
    # ingredient-ID cross-entropy.  The ingredient term is the main task.
    proportion_weight: float = 1.0
    # hierarchical vocabulary (#4) -- same as the JEPA
    coarse_size: int | None = None
    coarse_ids: list[int] | None = None


class CocktailMAE(nn.Module):
    """
    Masked-autoencoder over cocktail recipes.

    Encoder backbone identical to the JEPA's context encoder; the head is
    a reconstruction decoder instead of a latent predictor.  There is no
    target encoder, no EMA, no predictor, no SIGReg -- a masked
    autoencoder reconstructs inputs directly and cannot collapse.
    """

    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg

        # shared embedding layer -- the SAME hierarchical TokenEncoder the
        # JEPA uses, so the input representation is not a confound.
        self.tokens = TokenEncoder(
            cfg.vocab_size, cfg.prop_dim, cfg.d_model,
            coarse_size=cfg.coarse_size, coarse_ids=cfg.coarse_ids,
        )

        # encoder backbone -- same class and size as the JEPA encoder
        self.encoder = SetEncoder(
            d_model=cfg.d_model, n_layers=cfg.enc_layers,
            n_heads=cfg.enc_heads, dropout=cfg.dropout,
        )

        # reconstruction decoder: from the masked slot's contextualized
        # embedding, predict (a) the ingredient-ID logits and (b) the
        # proportion scalar.  This is the MAE's whole objective.
        self.ingredient_decoder = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.proportion_decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 4),
            nn.GELU(),
            nn.Linear(cfg.d_model // 4, 1),
        )

    def forward(self, batch: dict) -> dict:
        """
        Masked-reconstruction task on a batch.

        `batch` is what JEPAMaskCollator produces (the MAE reuses the same
        collator -- it needs exactly the same mask_index + target fields):
          ingredient_ids [B,L], proportions [B,L,P], pad_mask [B,L],
          mask_index [B], raw_proportion [B,L], target_proportion [B]

        Returns the loss dict (total + components).
        """
        ids = batch["ingredient_ids"]
        props = batch["proportions"]
        pad_mask = batch["pad_mask"]
        mask_index = batch["mask_index"]
        B = ids.shape[0]
        batch_idx = torch.arange(B, device=ids.device)

        # embed, mask the chosen slot, encode
        tokens = self.tokens(ids, props)
        masked_tokens = self.tokens.apply_mask(tokens, mask_index)
        enc = self.encoder(masked_tokens, pad_mask)            # [B, L, d]

        # the masked slot's contextualized embedding
        slot = enc[batch_idx, mask_index]                      # [B, d]

        # --- ingredient reconstruction: cross-entropy on the true ID ----
        ing_logits = self.ingredient_decoder(slot)             # [B, vocab]
        true_ids = ids[batch_idx, mask_index]                  # [B]
        ing_loss = F.cross_entropy(ing_logits, true_ids)

        # --- proportion reconstruction: masked MSE ----------------------
        prop_loss = self._proportion_loss(slot, batch, batch_idx, mask_index)

        total = ing_loss + self.cfg.proportion_weight * prop_loss
        return {
            "loss": total,
            "ingredient_loss": ing_loss,
            "proportion_loss": prop_loss,
        }

    def _proportion_loss(
        self,
        slot: torch.Tensor,             # [B, d]
        batch: dict,
        batch_idx: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked MSE on the reconstructed proportion scalar.

        Uses the exact proportion target the dataset already provides
        (batch['target_proportion'], a raw scalar, -1.0 where the masked
        slot has no parseable proportion) and supervises only the slots
        where it is known.  Returns 0 if no slot in the batch has one.
        """
        if "target_proportion" in batch:
            true_p = batch["target_proportion"].to(slot.dtype)
            known = true_p >= 0
        else:
            # fall back to the Fourier known-flag if the raw scalar is
            # absent (older batches) -- proportion is then not supervised
            return slot.new_zeros(())

        if known.sum() < 1:
            return slot.new_zeros(())

        pred_p = torch.sigmoid(self.proportion_decoder(slot).squeeze(-1))
        sq_err = (pred_p - true_p).pow(2) * known
        return sq_err.sum() / known.sum().clamp(min=1.0)

    def num_parameters(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "tokens": count(self.tokens),
            "encoder": count(self.encoder),
            "ingredient_decoder": count(self.ingredient_decoder),
            "proportion_decoder": count(self.proportion_decoder),
            "trainable_total": count(self),
        }


def build_mae(vocab_size: int, prop_dim: int, **overrides) -> CocktailMAE:
    """Convenience constructor, mirroring build_jepa."""
    cfg = MAEConfig(vocab_size=vocab_size, prop_dim=prop_dim, **overrides)
    return CocktailMAE(cfg)


def load_mae_checkpoint(path, map_location: str = "cpu") -> dict:
    """
    Load an MAE checkpoint and rebuild the CocktailMAE.

    train.checkpoint.save_checkpoint is generic (it stores asdict(cfg) +
    state_dict), so it writes MAE checkpoints fine -- but its
    load_checkpoint is hardwired to rebuild a CocktailJEPA from a
    JEPAConfig and cannot load an MAE.  This is the MAE counterpart,
    kept here so the MAE baseline is self-contained and the JEPA loader
    stays untouched.

    Returns {"model", "config", "step", "extra"} -- same shape as
    load_checkpoint's result, so the #43 table code can treat them alike.
    """
    import dataclasses

    import torch as _torch

    blob = _torch.load(path, map_location=map_location, weights_only=False)
    valid = {f.name for f in dataclasses.fields(MAEConfig)}
    config = {k: v for k, v in blob["config"].items() if k in valid}
    cfg = MAEConfig(**config)
    model = CocktailMAE(cfg)
    missing, unexpected = model.load_state_dict(blob["model_state"],
                                                strict=False)
    if missing:
        print(f"[mae checkpoint] absent keys: {list(missing)}")
    if unexpected:
        print(f"[mae checkpoint] unknown keys: {list(unexpected)}")
    return {
        "model": model,
        "config": cfg,
        "step": blob.get("step", 0),
        "extra": blob.get("extra", {}),
    }


# ---------------------------------------------------------------------------
# MAE energy -- the counterpart of energy.recipe_energy
# ---------------------------------------------------------------------------

@torch.no_grad()
def recipe_energy_mae(
    model: CocktailMAE,
    batch: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Energy for each recipe: mean masked-slot RECONSTRUCTION error.

    The MAE analog of energy.recipe_energy.  For a recipe of n
    ingredients, mask each slot in turn, measure how badly the decoder
    reconstructs that slot's ingredient (cross-entropy of the true ID
    under the predicted distribution), average over the n slots.  High
    reconstruction error = the slots do not predict one another =
    incoherent.

    `batch` is a plain (non-masking) _stack batch -- this function
    supplies its own deterministic per-slot masks, exactly like
    recipe_energy.  Returns FloatTensor [B].

    Note: energy here is the INGREDIENT reconstruction error only.  The
    proportion term is a training auxiliary; the energy mirrors the
    JEPA's choice of a single coherent error signal, and ingredient
    cross-entropy is the MAE's primary objective.
    """
    model.eval()
    ids = batch["ingredient_ids"].to(device)
    props = batch["proportions"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    n_ing = batch["n_ingredients"]
    B, L = ids.shape
    batch_idx = torch.arange(B, device=device)

    tokens = model.tokens(ids, props)

    err_sum = torch.zeros(B, device=device)
    slot_count = torch.zeros(B, device=device)

    max_n = int(n_ing.max().item())
    for slot in range(max_n):
        active = (slot < n_ing).to(device)
        if not active.any():
            break

        mask_index = torch.full((B,), slot, device=device, dtype=torch.long)
        masked_tokens = model.tokens.apply_mask(tokens, mask_index)
        enc = model.encoder(masked_tokens, pad_mask)
        slot_emb = enc[batch_idx, mask_index]                  # [B, d]

        logits = model.ingredient_decoder(slot_emb)            # [B, vocab]
        true_ids = ids[batch_idx, mask_index]                  # [B]
        # per-recipe cross-entropy at this slot
        err = F.cross_entropy(logits, true_ids, reduction="none")  # [B]
        err_sum += err * active
        slot_count += active.float()

    return err_sum / slot_count.clamp(min=1.0)


@torch.no_grad()
def mae_energy_over_loader(
    model: CocktailMAE,
    loader,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Energy for every recipe from a DataLoader -- mirrors
    energy.energy_over_loader so the #43 table scores MAE rows the same
    way as JEPA rows."""
    model.to(device)
    energies, ids = [], []
    for batch in loader:
        e = recipe_energy_mae(model, batch, device=device)
        energies.append(e.cpu())
        ids.extend(batch["recipe_id"])
    return torch.cat(energies), ids
