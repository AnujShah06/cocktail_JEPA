"""
jepa.py -- the full Joint-Embedding Predictive Architecture.

Ties together the three networks:
  * token encoder    -- shared embedding layer (ingredient + proportion)
  * context encoder  -- SetEncoder, sees the MASKED recipe, trained by SGD
  * target encoder   -- SetEncoder, sees the FULL recipe, an EMA copy of
                        the context encoder, NEVER trained by gradient
  * predictor        -- predicts the masked slot's contextualized latent

The forward pass implements the JEPA task:
  1. embed the recipe into tokens
  2. CONTEXT view: mask one slot, run the context encoder
  3. TARGET view : run the target encoder on the FULL recipe, take the
     contextualized embedding at the masked slot, DETACH it
  4. predictor: from the context + a query for the masked slot, predict
     that detached target latent

The EMA update of the target encoder is a separate explicit step
(`ema_update`) called by the Stage 3 training loop after each optimizer
step -- it is deliberately not hidden inside forward().
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cocktail_jepa.model.encoder import Predictor, SetEncoder
from cocktail_jepa.model.loss import jepa_loss
from cocktail_jepa.model.tokens import TokenEncoder


@dataclass
class JEPAConfig:
    """Architecture + loss hyperparameters for the JEPA."""
    vocab_size: int
    prop_dim: int
    d_model: int = 192
    enc_layers: int = 3
    enc_heads: int = 6
    pred_layers: int = 1
    pred_heads: int = 6
    dropout: float = 0.1
    ema_decay: float = 0.996
    var_weight: float = 0.5
    cov_weight: float = 0.04


class CocktailJEPA(nn.Module):
    """The full JEPA model: context + target encoders, predictor, loss."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        # shared embedding layer
        self.tokens = TokenEncoder(cfg.vocab_size, cfg.prop_dim, cfg.d_model)

        # context encoder -- trained by gradient descent
        self.context_encoder = SetEncoder(
            d_model=cfg.d_model, n_layers=cfg.enc_layers,
            n_heads=cfg.enc_heads, dropout=cfg.dropout,
        )

        # target encoder -- a deep copy, EMA-updated, gradient-free
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # predictor -- lower capacity
        self.predictor = Predictor(
            d_model=cfg.d_model, n_layers=cfg.pred_layers,
            n_heads=cfg.pred_heads, dropout=cfg.dropout,
        )

    # -- core forward ---------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Run the JEPA masked-prediction task on a batch.

        `batch` is what JEPAMaskCollator produces:
          ingredient_ids [B,L], proportions [B,L,P],
          pad_mask [B,L], mask_index [B]

        Returns the loss dict from jepa_loss plus the raw predicted /
        target latents (useful for Stage 4's energy function).
        """
        ids = batch["ingredient_ids"]
        props = batch["proportions"]
        pad_mask = batch["pad_mask"]
        mask_index = batch["mask_index"]
        B = ids.shape[0]
        batch_idx = torch.arange(B, device=ids.device)

        # 1. embed the full recipe into tokens
        tokens = self.tokens(ids, props)                  # [B, L, d]

        # 2. CONTEXT view: hide the masked slot, run the context encoder
        context_tokens = self.tokens.apply_mask(tokens, mask_index)
        context_emb = self.context_encoder(context_tokens, pad_mask)  # [B,L,d]

        # 3. TARGET view: full recipe through the target encoder, no grad
        with torch.no_grad():
            target_emb = self.target_encoder(tokens, pad_mask)        # [B,L,d]
        target_latent = target_emb[batch_idx, mask_index].detach()    # [B,d]

        # 4. predictor: query = the masked slot's context embedding
        query = context_emb[batch_idx, mask_index].unsqueeze(1)       # [B,1,d]
        predicted = self.predictor(query, context_emb, pad_mask)      # [B,d]

        # var/cov regularizer is computed on all REAL slot embeddings
        real_emb = context_emb[pad_mask]                  # [N, d]

        out = jepa_loss(
            predicted=predicted,
            target=target_latent,
            context_embeddings=real_emb,
            var_weight=self.cfg.var_weight,
            cov_weight=self.cfg.cov_weight,
        )
        out["predicted"] = predicted
        out["target"] = target_latent
        return out

    # -- EMA update -----------------------------------------------------
    @torch.no_grad()
    def ema_update(self, decay: float | None = None) -> None:
        """Move the target encoder toward the context encoder by EMA.

        Called by the Stage 3 training loop AFTER each optimizer step.
        target <- decay * target + (1 - decay) * context
        """
        d = self.cfg.ema_decay if decay is None else decay
        for tgt, src in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            tgt.mul_(d).add_(src, alpha=1.0 - d)
        # buffers (e.g. LayerNorm running stats) are copied outright
        for tgt, src in zip(self.target_encoder.buffers(),
                            self.context_encoder.buffers()):
            tgt.copy_(src)

    # -- convenience ----------------------------------------------------
    def num_parameters(self) -> dict[str, int]:
        """Parameter counts, for logging. Target encoder excluded -- it is
        not trained."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "tokens": count(self.tokens),
            "context_encoder": count(self.context_encoder),
            "predictor": count(self.predictor),
            "trainable_total": count(self),
        }


def build_jepa(vocab_size: int, prop_dim: int, **overrides) -> CocktailJEPA:
    """Convenience constructor used by the training script and tests."""
    cfg = JEPAConfig(vocab_size=vocab_size, prop_dim=prop_dim, **overrides)
    return CocktailJEPA(cfg)
