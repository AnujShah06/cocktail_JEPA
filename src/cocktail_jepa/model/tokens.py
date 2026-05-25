"""
tokens.py -- the input token layer of the JEPA.

Turns one ingredient slot into one model token.  Phase-1/2 fix #4 makes
the ingredient embedding HIERARCHICAL: a slot's ingredient contributes
TWO learned embeddings, summed --

    ingredient_token = embedding_fine(fine_id) + embedding_coarse(coarse_id)

  * the FINE embedding is specific to the exact ingredient ("bourbon" vs
    "scotch" are different fine ids -> different fine embeddings)
  * the COARSE embedding is shared across a whole category / spirit family
    ("bourbon" and "scotch" share the coarse id "whiskey" -> the SAME
    coarse embedding)

So bourbon and scotch differ only by their fine embedding, while sharing
the coarse "whiskey" signal -- the model sees "these are both whiskeys"
for free, and a rare fine token (a near-singleton brand-generic) still
gets a well-trained coarse embedding behind it.  This fixes the
granularity loss jepa-04 had, where bourbon and scotch collapsed together.

The fine-id -> coarse-id map comes from the hierarchical Vocabulary
(data/vocab.py, `coarse_ids`).  TokenEncoder stores it as a buffer
`coarse_of` so the coarse id for any fine id is a pure lookup -- no data
plumbing through the batch.

A slot also has a Fourier proportion, projected to model width and added,
gated by a learnable per-dimension `proportion_gate` (so the strong
ingredient signal does not swamp the weaker proportion signal).

The learned [MASK] embedding lives here too.  When a slot is masked for
the JEPA task, the context encoder must see [MASK] at that position
instead of the real token -- `apply_mask` does that substitution.

Nothing here is JEPA-specific logic; it is the shared embedding layer
both the context encoder and the target encoder are built on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cocktail_jepa.data.vocab import MASK_ID, PAD_ID


class TokenEncoder(nn.Module):
    """Ingredient (coarse+fine) + Fourier proportion -> a d_model-dim token."""

    def __init__(
        self,
        vocab_size: int,
        prop_dim: int,
        d_model: int = 192,
        coarse_size: int | None = None,
        coarse_ids: torch.Tensor | list[int] | None = None,
    ):
        """
        vocab_size  : number of FINE ids (incl. [PAD], [MASK]).
        coarse_size : number of COARSE ids (incl. coarse [PAD], [MASK]).
        coarse_ids  : LongTensor / list of length vocab_size; entry k is the
                      coarse id for fine id k.  Obtain it from
                      Vocabulary.coarse_ids (or .coarse_ids_tensor()).

        coarse_size and coarse_ids are optional ONLY so that an old
        checkpoint with no hierarchical vocabulary can still construct the
        module (it then runs in a degenerate single-coarse mode).  New
        training MUST pass both.
        """
        super().__init__()
        self.d_model = d_model

        # ---- fine ingredient embedding ----------------------------------
        # PAD_ID row is kept at zero and frozen.
        self.ingredient_emb = nn.Embedding(vocab_size, d_model,
                                           padding_idx=PAD_ID)

        # ---- coarse ingredient embedding --------------------------------
        if coarse_size is None or coarse_ids is None:
            # degenerate fallback: a single coarse id (besides PAD/MASK)
            # that every fine id maps to -- so an old non-hierarchical
            # checkpoint can still build.  Coarse signal is then a constant
            # bias, i.e. effectively disabled.
            coarse_size = 3  # [PAD], [MASK], one [JUNK]
            ids = [0, 1] + [2] * (vocab_size - 2)
            coarse_ids_t = torch.tensor(ids, dtype=torch.long)
        else:
            coarse_ids_t = (
                coarse_ids if torch.is_tensor(coarse_ids)
                else torch.tensor(coarse_ids, dtype=torch.long)
            )
            coarse_ids_t = coarse_ids_t.long()

        assert coarse_ids_t.numel() == vocab_size, (
            f"coarse_ids length {coarse_ids_t.numel()} must equal "
            f"vocab_size {vocab_size}"
        )
        self.coarse_size = coarse_size
        self.coarse_emb = nn.Embedding(coarse_size, d_model, padding_idx=PAD_ID)
        # fine-id -> coarse-id lookup; a buffer so it moves with .to(device)
        # and is saved in the checkpoint, but is NOT a trained parameter.
        self.register_buffer("coarse_of", coarse_ids_t, persistent=True)

        # ---- proportion ------------------------------------------------
        self.proportion_proj = nn.Linear(prop_dim, d_model)
        # learnable per-dimension gate on the proportion signal. The
        # ingredient embedding is a large, well-trained signal; without a
        # gate, summing ing + prop lets it swamp the (weaker) proportion
        # signal, leaving the model nearly blind to proportion structure.
        # Initialized to 1.0 so this starts identical to plain summation.
        self.proportion_gate = nn.Parameter(torch.ones(d_model))

        # ---- the learned [MASK] embedding, substituted at masked slots --
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def ingredient_embedding(self, ingredient_ids: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical ingredient embedding for a batch of fine ids.

        ingredient_ids : [B, L] long
        returns        : [B, L, d_model]  = fine_emb + coarse_emb
        """
        fine = self.ingredient_emb(ingredient_ids)             # [B, L, d]
        coarse_ids = self.coarse_of[ingredient_ids]            # [B, L]
        coarse = self.coarse_emb(coarse_ids)                   # [B, L, d]
        return fine + coarse

    def forward(
        self,
        ingredient_ids: torch.Tensor,   # [B, L]  long
        proportions: torch.Tensor,      # [B, L, P] float
    ) -> torch.Tensor:
        """Return token embeddings [B, L, d_model]."""
        ing = self.ingredient_embedding(ingredient_ids)        # [B, L, d]
        prop = self.proportion_proj(proportions)               # [B, L, d]
        return ing + self.proportion_gate * prop

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
