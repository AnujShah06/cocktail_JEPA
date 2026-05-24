"""
config.py -- single source of truth for paths, device, and run settings.

Design goal: the SAME code runs on the MacBook (mps) and a rented cloud
GPU (cuda) with no edits. Anything environment-specific is read from an
environment variable with a sensible local default, so moving to the
cloud means setting env vars, not changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_device() -> str:
    """cuda on a cloud GPU, mps on Apple Silicon, cpu as last resort."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# project root = two levels up from this file (src/cocktail_jepa/config.py)
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    """All filesystem locations. Override the roots via env vars on cloud."""
    root: Path = ROOT
    # corpus location -- on cloud, point COCKTAIL_DATA_DIR at the synced bucket
    corpus: Path = field(
        default_factory=lambda: Path(os.environ.get("COCKTAIL_DATA_DIR",
                                                     ROOT / "corpus"))
    )
    # run artifacts -- on cloud, point COCKTAIL_RUN_DIR at persistent storage
    runs: Path = field(
        default_factory=lambda: Path(os.environ.get("COCKTAIL_RUN_DIR",
                                                    ROOT / "runs"))
    )

    @property
    def recipes(self) -> Path:
        return self.corpus / "recipes.jsonl"

    @property
    def vocabulary(self) -> Path:
        return self.corpus / "vocabulary.json"


@dataclass
class Config:
    """Top-level config. Stages will extend this with their own settings."""
    device: str = field(default_factory=_detect_device)
    seed: int = 42
    paths: Paths = field(default_factory=Paths)

    # experiment tracking -- off by default; flip on (or set COCKTAIL_WANDB=1)
    # when Stage 3 training begins.
    use_wandb: bool = field(
        default_factory=lambda: os.environ.get("COCKTAIL_WANDB", "0") == "1"
    )
    wandb_project: str = "cocktail-jepa"


# a ready-to-import default; stages can also build their own Config
CONFIG = Config()
