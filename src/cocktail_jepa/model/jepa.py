"""
jepa.py -- the full Joint-Embedding Predictive Architecture.

Ties together the networks:
  * token encoder    -- shared embedding layer (hierarchical ingredient
                        coarse+fine, plus proportion)
  * context encoder  -- SetEncoder, sees the MASKED recipe, trained by SGD
  * target encoder   -- SetEncoder, sees the FULL recipe, an EMA copy of
                        the context encoder, NEVER trained by gradient
  * predictor        -- predicts the masked slot's contextualized latent
  * proportion head  -- (Phase-2 fix #13) predicts the masked slot's
                        proportion scalar from the predicted latent

The forward pass implements the JEPA task:
  1. embed the recipe into tokens (hierarchical coarse+fine -- #4)
  2. CONTEXT view: mask one slot, run the context encoder
  3. TARGET view : run the target encoder on the FULL recipe, take the
     contextualized embedding at the masked slot, DETACH it
  4. predictor: from the context + a query for the masked slot, predict
     that detached target latent
  5. proportion head: from the predicted latent, regress the masked
     slot's true proportion -- an auxiliary objective (#13)

PHASE-2 CHANGES
---------------
#18  the loss is now prediction loss + SIGReg (one regularizer, one
     weight `sigreg_weight`) -- see model/loss.py.  The old var_weight /
     cov_weight config fields are gone.
#13  a proportion auxiliary head + loss.  The JEPA latent prediction
     alone did not reward proportion sensitivity (jepa-04 scored only
     ~0.57 on the proportion-scramble perturbation).  The aux head
     regresses the masked slot's proportion FROM THE PREDICTED LATENT --
     not from the slot's own input proportion (that would be circular,
     since the predictor's query already carries it).  Forcing a small
     head to recover proportion from the latent pushes proportion
     structure into the representation.
#4   TokenEncoder is hierarchical; JEPAConfig carries the coarse-vocab
     size and the fine->coarse id map needed to build it.

The EMA update of the target encoder is a separate explicit step
(`ema_update`) called by the training loop after each optimizer step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # --- #18: SIGReg single-regularizer objective --------------------
    # one trade-off hyperparameter replaces the old var_weight + cov_weight
    sigreg_weight: float = 1.0
    sigreg_projections: int = 64

    # --- #13: proportion auxiliary loss ------------------------------
    # weight on the auxiliary proportion-regression term; 0.0 disables it
    proportion_aux_weight: float = 0.5

    # --- #4: hierarchical vocabulary ---------------------------------
    # coarse-vocabulary size and the fine-id -> coarse-id map.  Defaults
    # leave the coarse level degenerate (single bucket) so an old
    # non-hierarchical checkpoint config still constructs; real training
    # sets both from the Vocabulary.
    coarse_size: int | None = None
    coarse_ids: list[int] | None = None


class CocktailJEPA(nn.Module):
    """The full JEPA model: context + target encoders, predictor, heads."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        # shared embedding layer -- hierarchical (coarse + fine), #4
        self.tokens = TokenEncoder(
            cfg.vocab_size, cfg.prop_dim, cfg.d_model,
            coarse_size=cfg.coarse_size,
            coarse_ids=cfg.coarse_ids,
        )

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

        # proportion auxiliary head -- #13.  Small on purpose: a near-linear
        # readout, so a LOW aux loss really means the proportion is in the
        # LATENT, not that a deep head reconstructed it.  Outputs one scalar
        # (a logit; a sigmoid maps it to a proportion in (0, 1)).
        self.proportion_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 4),
            nn.GELU(),
            nn.Linear(cfg.d_model // 4, 1),
        )

    # -- core forward ---------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Run the JEPA masked-prediction task on a batch.

        `batch` is what JEPAMaskCollator produces:
          ingredient_ids [B,L], proportions [B,L,P],
          pad_mask [B,L], mask_index [B]

        Returns the loss dict (total loss + components) plus the raw
        predicted / target latents (used by Stage 4's energy function).
        """
        ids = batch["ingredient_ids"]
        props = batch["proportions"]
        pad_mask = batch["pad_mask"]
        mask_index = batch["mask_index"]
        B = ids.shape[0]
        batch_idx = torch.arange(B, device=ids.device)

        # 1. embed the full recipe into tokens (hierarchical coarse+fine)
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

        # SIGReg regularizer is computed on all REAL slot embeddings
        real_emb = context_emb[pad_mask]                  # [N, d]

        out = jepa_loss(
            predicted=predicted,
            target=target_latent,
            context_embeddings=real_emb,
            sigreg_weight=self.cfg.sigreg_weight,
            sigreg_projections=self.cfg.sigreg_projections,
        )

        # 5. proportion auxiliary loss (#13) -----------------------------
        # regress the masked slot's TRUE proportion from the predicted
        # latent.  The true proportion and a "known" flag are recovered
        # from the Fourier proportion encoding the dataset already built:
        # by construction the last channel is the known-flag, and channel
        # 1 is cos(pi * p)  (see vocab.fourier_proportion_encoding) -- but
        # rather than invert the encoding we read the proportion straight
        # from the batch if present, else fall back to the cos channel.
        aux = self._proportion_aux_loss(predicted, props, mask_index,
                                        batch_idx, batch)
        out["proportion_aux"] = aux
        out["loss"] = out["loss"] + self.cfg.proportion_aux_weight * aux

        out["predicted"] = predicted
        out["target"] = target_latent
        return out

    def _proportion_aux_loss(
        self,
        predicted: torch.Tensor,        # [B, d]
        props: torch.Tensor,            # [B, L, P]  Fourier proportion enc.
        mask_index: torch.Tensor,       # [B]
        batch_idx: torch.Tensor,        # [B]
        batch: dict,
    ) -> torch.Tensor:
        """
        Auxiliary loss: predict the masked slot's proportion scalar from
        the predicted latent.  Slots whose proportion is UNKNOWN (the
        recipe had no parseable quantity) are excluded -- the Fourier
        encoding's last channel is the known-flag, 1.0 iff a proportion
        was supplied.  Returns a scalar; 0 if no slot in the batch has a
        known proportion.
        """
        # the masked slot's Fourier proportion vector: [B, P]
        masked_prop = props[batch_idx, mask_index]            # [B, P]
        known = masked_prop[:, -1]                            # [B]  0/1 flag

        # recover the proportion scalar.  The dataset may also pass the raw
        # scalar via batch["target_proportion"]; prefer that when present,
        # else reconstruct from the encoding's first cos channel:
        #   channel index 1 is cos(pi * p)  ->  p = acos(.)/pi  for p in [0,1]
        if "target_proportion" in batch:
            true_p = batch["target_proportion"].to(predicted.dtype)
        else:
            cos_pi_p = masked_prop[:, 1].clamp(-1.0, 1.0)     # cos(pi*p)
            true_p = torch.acos(cos_pi_p) / torch.pi          # [B] in [0,1]

        if known.sum() < 1:
            return predicted.new_zeros(())

        pred_logit = self.proportion_head(predicted).squeeze(-1)  # [B]
        pred_p = torch.sigmoid(pred_logit)

        # masked MSE: supervise only slots with a known proportion
        sq_err = (pred_p - true_p).pow(2) * known
        return sq_err.sum() / known.sum().clamp(min=1.0)

    # -- EMA update -----------------------------------------------------
    @torch.no_grad()
    def ema_update(self, decay: float | None = None) -> None:
        """Move the target encoder toward the context encoder by EMA.

        Called by the training loop AFTER each optimizer step.
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
            "proportion_head": count(self.proportion_head),
            "trainable_total": count(self),
        }


def build_jepa(vocab_size: int, prop_dim: int, **overrides) -> CocktailJEPA:
    """Convenience constructor used by the training script and tests."""
    cfg = JEPAConfig(vocab_size=vocab_size, prop_dim=prop_dim, **overrides)
    return CocktailJEPA(cfg)
