"""
dataset.py -- the CocktailDataset and the JEPA masking collate layer.

Design (decided up front):
  * The Dataset yields PLAIN recipes as fixed-width tensors -- no masking.
    Its single job is recipe -> tensors.
  * Masking is a SEPARATE collate function. Stage 3 training uses random
    masking; Stage 4 energy evaluation uses a deterministic mask. Keeping
    masking out of the Dataset means the same data serves both, just by
    swapping the collate function.
  * Masking marks WHICH slot is hidden (an index). The actual [MASK]-token
    embedding substitution happens inside the model in Stage 2 -- the
    collate only produces indices and the padding mask.

A recipe of n ingredients becomes:
  ingredient_ids : LongTensor [max_len]      ingredient id per slot, [PAD] filled
  proportions    : FloatTensor [max_len, P]  Fourier proportion encoding per slot
  pad_mask       : BoolTensor  [max_len]     True where the slot is real
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cocktail_jepa.data.vocab import (
    PAD_ID,
    Vocabulary,
    fourier_proportion_encoding,
    proportion_encoding_dim,
)


class CocktailDataset(Dataset):
    """Plain recipe dataset. Yields fixed-width padded tensors, no masking."""

    def __init__(
        self,
        recipes: list[dict],
        vocab: Vocabulary,
        max_len: int = 12,
        n_frequencies: int = 6,
    ):
        # keep only recipes that fit; >max_len ingredients are rare and
        # truncating them would distort proportions, so we drop them.
        self.recipes = [r for r in recipes if 2 <= len(r["ingredients"]) <= max_len]
        self.vocab = vocab
        self.max_len = max_len
        self.n_frequencies = n_frequencies
        self.prop_dim = proportion_encoding_dim(n_frequencies)

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> dict:
        recipe = self.recipes[idx]
        ings = recipe["ingredients"]
        n = len(ings)

        ids = np.full(self.max_len, PAD_ID, dtype=np.int64)
        props = np.zeros((self.max_len, self.prop_dim), dtype=np.float32)
        pad_mask = np.zeros(self.max_len, dtype=bool)

        for i, ing in enumerate(ings):
            ids[i] = self.vocab.encode(ing["ingredient"])
            props[i] = fourier_proportion_encoding(
                ing.get("proportion"), self.n_frequencies
            )
            pad_mask[i] = True

        return {
            "ingredient_ids": torch.from_numpy(ids),
            "proportions": torch.from_numpy(props),
            "pad_mask": torch.from_numpy(pad_mask),
            "n_ingredients": n,
            "recipe_id": recipe.get("recipe_id", ""),
        }


def _stack(batch: list[dict]) -> dict:
    """Stack a list of dataset items into batched tensors."""
    return {
        "ingredient_ids": torch.stack([b["ingredient_ids"] for b in batch]),
        "proportions": torch.stack([b["proportions"] for b in batch]),
        "pad_mask": torch.stack([b["pad_mask"] for b in batch]),
        "n_ingredients": torch.tensor([b["n_ingredients"] for b in batch]),
        "recipe_id": [b["recipe_id"] for b in batch],
    }


class JEPAMaskCollator:
    """
    Collate function that adds JEPA masking on top of a batch.

    For each recipe it picks one real ingredient slot to be the masked
    target. It does NOT alter ingredient_ids -- the model will substitute
    the [MASK] embedding at the chosen index in Stage 2. The collator only
    reports which index is masked.

    Adds to the batch:
      mask_index : LongTensor [B]   the masked slot per recipe

    deterministic=True always masks the same slot (the last real
    ingredient) -- used by Stage 4 so energy scores are reproducible.
    deterministic=False masks a uniformly random real slot -- used by
    Stage 3 training.
    """

    def __init__(self, deterministic: bool = False, seed: int = 0):
        self.deterministic = deterministic
        self._rng = random.Random(seed)

    def __call__(self, batch: list[dict]) -> dict:
        out = _stack(batch)
        B = out["ingredient_ids"].shape[0]
        n_ing = out["n_ingredients"]

        mask_index = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            n = int(n_ing[b].item())
            if self.deterministic:
                mask_index[b] = n - 1  # last real slot, stable
            else:
                mask_index[b] = self._rng.randint(0, n - 1)
        out["mask_index"] = mask_index
        return out


def load_recipes(path: str | Path) -> list[dict]:
    """Read a recipes .jsonl file into a list of dicts."""
    return [json.loads(line) for line in open(path, encoding="utf-8")]
