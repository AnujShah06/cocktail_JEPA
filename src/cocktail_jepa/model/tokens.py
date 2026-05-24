"""
tokens.py -- the input token layer of the JEPA.

Turns one ingredient slot into one model token. A slot has two parts:
  * an ingredient id      -> looked up in a learned embedding table
  * a Fourier proportion  -> projected to the model width and added

The learned [MASK] embedding also lives here. When a slot is masked for
the JEPA task, the context encoder must see [MASK] at that position
instead of the real token -- this module provides that substitution via
`apply_mask`.

Nothing here is JEPA-specific logic; it is the shared embedding layer
both the context encoder and the target encoder are built on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cocktail_jepa.data.vocab import MASK_ID, PAD_ID


class TokenEncoder(nn.Module):
    """Ingredient id + Fourier proportion -> a d_model-dim token."""

    def __init__(self, vocab_size: int, prop_dim: int, d_model: int = 192):
        super().__init__()
        self.d_model = d_model
        # ingredient embedding table; PAD_ID row is kept at zero and frozen
        self.ingredient_emb = nn.Embedding(vocab_size, d_model,
                                           padding_idx=PAD_ID)
        # project the Fourier proportion vector into model space
        self.proportion_proj = nn.Linear(prop_dim, d_model)
        # the learned [MASK] embedding, substituted at masked slots
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        ingredient_ids: torch.Tensor,   # [B, L]  long
        proportions: torch.Tensor,      # [B, L, P] float
    ) -> torch.Tensor:
        """Return token embeddings [B, L, d_model]."""
        ing = self.ingredient_emb(ingredient_ids)          # [B, L, d]
        prop = self.proportion_proj(proportions)           # [B, L, d]
        return ing + prop

    def apply_mask(
        self,
        tokens: torch.Tensor,           # [B, L, d_model]
        mask_index: torch.Tensor,       # [B]  long, the slot to hide
    ) -> torch.Tensor:
        """Replace the token at mask_index (per recipe) with [MASK].

        Used to build the CONTEXT view: the context encoder sees the
        recipe with one slot hidden. The target encoder, by contrast,
        sees the full unmasked recipe.
        """
        out = tokens.clone()
        B = tokens.shape[0]
        batch_idx = torch.arange(B, device=tokens.device)
        out[batch_idx, mask_index] = self.mask_token
        return out
