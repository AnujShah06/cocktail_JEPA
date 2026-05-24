"""
encoder.py -- the set-Transformer encoder and the predictor.

SetEncoder
  A small Transformer encoder with NO positional encoding -- a cocktail is
  an unordered set, so position carries no meaning and adding it would
  inject spurious structure. Self-attention provides permutation-
  equivariant mixing: each ingredient slot is represented in the context
  of its companions. Padding slots are masked out of attention.

  Both the context encoder and the target encoder are SetEncoder
  instances (the target one is an EMA copy -- see jepa.py).

Predictor
  Deliberately LOWER capacity than the encoder. Given the context
  encoder's output and a query marking the masked slot, it predicts that
  slot's contextualized embedding. Keeping the predictor weak forces the
  representational work into the encoder rather than letting the predictor
  absorb the task -- one of the three anti-collapse mechanisms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """Permutation-equivariant Transformer encoder over a set of tokens."""

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm -- more stable to train
        )
        # pre-norm is chosen for training stability; this disables an
        # unrelated nested-tensor optimization, so silence its warning.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,       # [B, L, d_model]
        pad_mask: torch.Tensor,     # [B, L]  True where real, False where pad
    ) -> torch.Tensor:
        """Return contextualized embeddings [B, L, d_model].

        nn.Transformer expects src_key_padding_mask True where a position
        should be IGNORED, which is the inverse of our pad_mask, hence the
        ~ negation.
        """
        h = self.encoder(tokens, src_key_padding_mask=~pad_mask)
        return self.norm(h)


class Predictor(nn.Module):
    """Lower-capacity network: context + query slot -> predicted latent.

    The query for the masked slot is its OWN proportion token (the model
    is told the size of the slot it must fill, but not the ingredient).
    The predictor attends from that query over the context embeddings.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 1,        # << shallow on purpose
        n_heads: int = 6,
        ff_mult: int = 2,         # << narrow on purpose
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,        # [B, 1, d_model]  the masked-slot query
        context: torch.Tensor,      # [B, L, d_model]  context-encoder output
        pad_mask: torch.Tensor,     # [B, L]  True where real
    ) -> torch.Tensor:
        """Return the predicted latent for the masked slot: [B, d_model]."""
        h = query
        for layer in self.layers:
            h = layer(h, context, memory_key_padding_mask=~pad_mask)
        h = self.norm(h)
        return h.squeeze(1)         # [B, d_model]
