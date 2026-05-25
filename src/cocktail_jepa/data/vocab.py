"""
vocab.py -- ingredient vocabulary and input encoding.

Two jobs:
  1. Vocabulary: map every canonical ingredient string <-> an integer id.
     Reserves id 0 for [PAD] and id 1 for [MASK] so the model has stable
     special tokens; real ingredients start at id 2.
  2. Proportion encoding: turn a scalar proportion in [0, 1] into a smooth
     fixed-width Fourier feature vector, so the network sees magnitude at
     multiple frequencies rather than one raw number it must learn to scale.

HIERARCHICAL VOCABULARY (Phase-1 fix #4)
----------------------------------------
Every ingredient now carries TWO ids:
  * a FINE id   -- the specific canonical ingredient ("bourbon", "scotch")
  * a COARSE id -- its category-or-spirit-family ("whiskey" for both of the
                   above; "juice", "sweetener", ... for non-spirits)
The model token becomes  embedding(coarse) + embedding(fine)  -- the coarse
embedding is shared across every fine token in the family, so bourbon and
scotch are distinct fine tokens that nonetheless share a "whiskey" signal.

`Vocabulary` exposes `coarse_ids` -- a list indexed by fine id giving the
coarse id for that fine token (with [PAD]/[MASK] mapping to coarse
[PAD]/[MASK]).  `model/tokens.py` reads this to drive the second embedding.

BACKWARD COMPATIBILITY
----------------------
Old vocabulary.json files (pre-#4) have no "coarse_vocab" and their
ingredient entries have no "coarse" field.  `from_file` detects this and
builds a degenerate coarse vocabulary -- a single "[JUNK]" coarse token
that every fine ingredient maps to -- so old files still load.  A model
trained on the new vocabulary should be retrained, not loaded against an
old file, but the loader will not crash.

This module produces NO masking and NO tensors-per-recipe -- it is the
lookup layer the Dataset builds on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
JUNK_TOKEN = "[JUNK]"
PAD_ID = 0
MASK_ID = 1
N_SPECIAL = 2  # ids 0,1 reserved; real ingredients start at 2

# coarse ids 0,1 mirror the fine special tokens
COARSE_PAD_ID = 0
COARSE_MASK_ID = 1


class Vocabulary:
    """
    Bidirectional ingredient <-> id map, built from vocabulary.json.

    Holds BOTH levels of the hierarchical vocabulary:
      * the fine level   (id_to_token / token_to_id)  -- specific ingredient
      * the coarse level (coarse_id_to_token / ...)    -- category|family
      * coarse_ids[fine_id] -> coarse_id               -- the link between them
    """

    def __init__(
        self,
        ingredients: list[str],
        coarse_vocab: list[str] | None = None,
        fine_to_coarse: list[int] | None = None,
    ):
        # ---- fine level --------------------------------------------------
        # ingredients: canonical strings; order only affects id assignment.
        self.id_to_token: list[str] = [PAD_TOKEN, MASK_TOKEN] + list(ingredients)
        self.token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self.id_to_token)
        }

        # ---- coarse level ------------------------------------------------
        if coarse_vocab is None or fine_to_coarse is None:
            # degenerate fallback (old file / ad-hoc construction): a single
            # [JUNK] coarse token that every real ingredient maps to.
            self.coarse_id_to_token: list[str] = [
                PAD_TOKEN, MASK_TOKEN, JUNK_TOKEN
            ]
            junk_id = 2
            self.coarse_ids: list[int] = (
                [COARSE_PAD_ID, COARSE_MASK_ID]
                + [junk_id] * len(ingredients)
            )
        else:
            self.coarse_id_to_token = list(coarse_vocab)
            # fine_to_coarse is given for the real ingredients only; prepend
            # the two special-token mappings so it is indexed by fine id.
            self.coarse_ids = (
                [COARSE_PAD_ID, COARSE_MASK_ID] + list(fine_to_coarse)
            )

        self.coarse_token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self.coarse_id_to_token)
        }

        # internal consistency: one coarse id per fine id
        assert len(self.coarse_ids) == len(self.id_to_token), (
            "coarse_ids must be parallel to id_to_token"
        )

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def n_ingredients(self) -> int:
        """Count of real ingredients, excluding [PAD] and [MASK]."""
        return len(self.id_to_token) - N_SPECIAL

    @property
    def coarse_size(self) -> int:
        """Total coarse vocabulary size, including coarse [PAD]/[MASK]."""
        return len(self.coarse_id_to_token)

    def encode(self, ingredient: str) -> int:
        """Ingredient string -> fine id. Unknown ingredients map to [MASK] id
        (id 1) as a safe fallback; in practice every recipe ingredient is
        in-vocabulary because the vocab was built from the same corpus."""
        return self.token_to_id.get(ingredient, MASK_ID)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return MASK_TOKEN

    def coarse_of(self, fine_id: int) -> int:
        """Coarse id for a fine id. Out-of-range ids map to coarse [MASK]."""
        if 0 <= fine_id < len(self.coarse_ids):
            return self.coarse_ids[fine_id]
        return COARSE_MASK_ID

    def coarse_ids_tensor(self):
        """
        The fine-id -> coarse-id map as a torch LongTensor, ready to register
        as a model buffer.  Imported lazily so this module has no hard torch
        dependency (vocab building runs in plain numpy environments).
        """
        import torch
        return torch.tensor(self.coarse_ids, dtype=torch.long)

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocabulary":
        """
        Load from the corpus vocabulary.json.

        New (#4) schema:
          {"size", "coarse_size", "coarse_vocab": [...],
           "ingredients": [{"name","count","category","coarse","coarse_id"}]}
        Old schema (pre-#4):
          {"size", "ingredients": [{"name","count","category"}]}
        Old files have no coarse information; they load via the degenerate
        single-[JUNK]-coarse fallback so legacy checkpoints still run.
        """
        obj = json.load(open(path, encoding="utf-8"))
        names = [item["name"] for item in obj["ingredients"]]

        if "coarse_vocab" in obj and all(
            "coarse_id" in item for item in obj["ingredients"]
        ):
            coarse_vocab = obj["coarse_vocab"]
            fine_to_coarse = [item["coarse_id"] for item in obj["ingredients"]]
            return cls(names, coarse_vocab, fine_to_coarse)

        # old file: no coarse level -> degenerate fallback
        return cls(names)


def fourier_proportion_encoding(
    proportion: float | None,
    n_frequencies: int = 6,
) -> np.ndarray:
    """
    Encode a proportion scalar as a Fourier feature vector.

    A raw proportion (e.g. 0.33) is a single number the network would have
    to learn to interpret across scales. Instead we expand it into
    sin/cos pairs at geometrically increasing frequencies -- the standard
    positional-encoding trick applied to a continuous quantity. This gives
    the model a smooth, high-resolution representation of magnitude.

    Output width = 2 * n_frequencies + 1:
      - 2*n_frequencies  sin/cos values
      - 1 "known" flag   (1.0 if a proportion was supplied, 0.0 if missing)

    A missing proportion (recipe had no parseable quantities) yields a
    zero vector with the known-flag off, so the model can tell the
    difference between "proportion is 0" and "proportion unknown".
    """
    width = 2 * n_frequencies + 1
    out = np.zeros(width, dtype=np.float32)
    if proportion is None:
        return out  # all zeros, known-flag (last entry) stays 0
    p = float(proportion)
    for k in range(n_frequencies):
        freq = math.pi * (2 ** k)  # pi, 2pi, 4pi, ...
        out[2 * k] = math.sin(freq * p)
        out[2 * k + 1] = math.cos(freq * p)
    out[-1] = 1.0  # known flag
    return out


def proportion_encoding_dim(n_frequencies: int = 6) -> int:
    """Width of the Fourier proportion vector -- handy for model config."""
    return 2 * n_frequencies + 1
