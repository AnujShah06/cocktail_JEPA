"""
checkpoint.py -- save and load JEPA training state.

A checkpoint captures everything needed to (a) resume training or
(b) hand a trained model to Stage 4. That means model weights, optimizer
state, scheduler state, the step counter, and the JEPAConfig that built
the model (so it can be reconstructed without guessing hyperparameters).

Checkpoints go under runs/ -- git-ignored; on cloud they land in the
COCKTAIL_RUN_DIR persistent-storage path via config.py.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from cocktail_jepa.model.jepa import CocktailJEPA, JEPAConfig


def save_checkpoint(
    path: str | Path,
    model: CocktailJEPA,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    step: int = 0,
    extra: dict | None = None,
) -> Path:
    """Write a checkpoint. `extra` may hold metrics, e.g. best val loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model_state": model.state_dict(),
        "config": asdict(model.cfg),
        "step": step,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "extra": extra or {},
    }
    torch.save(blob, path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> dict:
    """
    Load a checkpoint and rebuild the model.

    Returns a dict: {"model", "config", "step", "optimizer_state",
    "scheduler_state", "extra"}. The model is reconstructed from the saved
    JEPAConfig and has the saved weights loaded -- ready for Stage 4 to
    use directly, or for the loop to resume from.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = JEPAConfig(**blob["config"])
    model = CocktailJEPA(cfg)
    model.load_state_dict(blob["model_state"])
    return {
        "model": model,
        "config": cfg,
        "step": blob.get("step", 0),
        "optimizer_state": blob.get("optimizer_state"),
        "scheduler_state": blob.get("scheduler_state"),
        "extra": blob.get("extra", {}),
    }
