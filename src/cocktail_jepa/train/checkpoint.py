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

    # JEPAConfig's fields change across versions (#18 removed var_weight /
    # cov_weight; #4/#13/#18 added sigreg_*, proportion_aux_weight,
    # coarse_*).  A checkpoint saved by an older version carries config
    # keys this JEPAConfig no longer accepts, and would crash JEPAConfig(
    # **config).  Filter the saved config down to the fields the current
    # JEPAConfig actually declares; any field the old checkpoint lacks
    # keeps the current default (e.g. coarse_size=None -> the TokenEncoder
    # degenerate single-coarse mode, sigreg_weight its default).
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(JEPAConfig)}
    raw_config = blob["config"]
    dropped = sorted(set(raw_config) - valid_fields)
    config = {k: v for k, v in raw_config.items() if k in valid_fields}
    if dropped:
        print(f"[checkpoint] ignoring obsolete config keys: {dropped}")
    cfg = JEPAConfig(**config)
    model = CocktailJEPA(cfg)
    # strict=False so checkpoints saved before a parameter was added
    # (e.g. the proportion_gate) still load -- any missing parameter keeps
    # its default initialization, which for the gate is 1.0 (a no-op).
    missing, unexpected = model.load_state_dict(blob["model_state"],
                                                strict=False)
    if missing:
        print(f"[checkpoint] using defaults for absent keys: {list(missing)}")
    if unexpected:
        print(f"[checkpoint] ignoring unknown keys: {list(unexpected)}")
    return {
        "model": model,
        "config": cfg,
        "step": blob.get("step", 0),
        "optimizer_state": blob.get("optimizer_state"),
        "scheduler_state": blob.get("scheduler_state"),
        "extra": blob.get("extra", {}),
    }
