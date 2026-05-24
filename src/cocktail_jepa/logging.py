"""
logging.py -- experiment tracking, wrapped so the codebase never touches
W&B directly.

Two modes, one interface:
  * use_wandb=True  -> metrics stream to Weights & Biases
  * use_wandb=False -> the same calls become cheap no-ops + console prints

Every stage calls `logger.log({...})` and `logger.summary(...)` without
caring which mode is active. This means Stage 1 plumbing runs with no
network dependency, and Stage 3 training gets full W&B tracking just by
flipping the flag (or setting COCKTAIL_WANDB=1).

Usage:
    from cocktail_jepa.logging import get_logger
    logger = get_logger(run_name="stage3-jepa", config={...})
    logger.log({"loss": 0.42, "encoder_rank": 87}, step=10)
    logger.summary(final_auroc=0.91)
    logger.finish()
"""

from __future__ import annotations

import json
from typing import Any

from cocktail_jepa.config import CONFIG


class Logger:
    """Unified experiment logger. Talks to W&B, or no-ops to the console."""

    def __init__(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        use_wandb: bool | None = None,
        tags: list[str] | None = None,
    ):
        # explicit arg wins; otherwise fall back to global config
        self.use_wandb = CONFIG.use_wandb if use_wandb is None else use_wandb
        self.run_name = run_name or "run"
        self._wandb = None

        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=CONFIG.wandb_project,
                    name=run_name,
                    config=config or {},
                    tags=tags or [],
                    dir=str(CONFIG.paths.runs),
                )
                print(f"[logger] W&B active -- project '{CONFIG.wandb_project}', "
                      f"run '{run_name}'")
            except Exception as e:
                # never let a logging failure kill a training run
                print(f"[logger] W&B unavailable ({e}); falling back to console")
                self.use_wandb = False

        if not self.use_wandb:
            print(f"[logger] console mode -- run '{self.run_name}'")
            if config:
                print(f"[logger] config: {json.dumps(config, default=str)}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of metrics at an optional step."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            prefix = f"[step {step}] " if step is not None else ""
            body = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(f"{prefix}{body}")

    def summary(self, **kwargs: Any) -> None:
        """Record final/summary values (best metric, final AUROC, ...)."""
        if self.use_wandb and self._wandb is not None:
            for k, v in kwargs.items():
                self._wandb.summary[k] = v
        else:
            body = "  ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[summary] {body}")

    def watch(self, model: Any) -> None:
        """Track gradients/parameters of a model (W&B only; no-op otherwise)."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.watch(model, log="all", log_freq=100)

    def finish(self) -> None:
        """Close the run cleanly."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()


def get_logger(
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    use_wandb: bool | None = None,
    tags: list[str] | None = None,
) -> Logger:
    """Convenience constructor -- the standard entry point for every stage."""
    return Logger(run_name=run_name, config=config, use_wandb=use_wandb, tags=tags)
