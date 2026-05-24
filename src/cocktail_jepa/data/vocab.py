"""
vocab.py -- ingredient vocabulary and input encoding.

Two jobs:
  1. Vocabulary: map every canonical ingredient string <-> an integer id.
     Reserves id 0 for [PAD] and id 1 for [MASK] so the model has stable
     special tokens; real ingredients start at id 2.
  2. Proportion encoding: turn a scalar proportion in [0, 1] into a smooth
     fixed-width Fourier feature vector, so the network sees magnitude at
     multiple frequencies rather than one raw number it must learn to scale.

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
PAD_ID = 0
MASK_ID = 1
N_SPECIAL = 2  # ids 0,1 reserved; real ingredients start at 2


class Vocabulary:
    """Bidirectional ingredient <-> id map, built from vocabulary.json."""

    def __init__(self, ingredients: list[str]):
        # ingredients: canonical strings, most-frequent-first is fine but
        # order only affects id assignment, nothing else.
        self.id_to_token: list[str] = [PAD_TOKEN, MASK_TOKEN] + list(ingredients)
        self.token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self.id_to_token)
        }

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def n_ingredients(self) -> int:
        """Count of real ingredients, excluding [PAD] and [MASK]."""
        return len(self.id_to_token) - N_SPECIAL

    def encode(self, ingredient: str) -> int:
        """Ingredient string -> id. Unknown ingredients map to [MASK] id
        (id 1) as a safe fallback; in practice every recipe ingredient is
        in-vocabulary because the vocab was built from the same corpus."""
        return self.token_to_id.get(ingredient, MASK_ID)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return MASK_TOKEN

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocabulary":
        """Load from the corpus vocabulary.json (the file build_corpus.py
        wrote: {"size": N, "ingredients": [{"name","count","category"},...]})."""
        obj = json.load(open(path, encoding="utf-8"))
        names = [item["name"] for item in obj["ingredients"]]
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
