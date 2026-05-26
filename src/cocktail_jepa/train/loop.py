"""
loop.py -- the JEPA training loop.

Implements exactly what the brief's Phase 2 specified: AdamW with cosine
learning-rate decay and a short warmup, gradient clipping, and the EMA
update of the target encoder after each optimizer step. Validation runs
periodically. Collapse diagnostics are computed and logged every run --
log-only, never auto-stop (decided).

This module does NOT touch the energy function, generation, or transfer
-- those are Stages 4-6. Its single job is to train the JEPA and surface
whether it is collapsing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.model.jepa import CocktailJEPA
from cocktail_jepa.train.checkpoint import save_checkpoint
from cocktail_jepa.train.diagnostics import collapse_report


@dataclass
class TrainConfig:
    """Training-loop hyperparameters (brief's Phase 2 spec)."""
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    grad_clip: float = 1.0
    val_every: int = 1            # run validation every N epochs
    diag_every: int = 1          # compute collapse diagnostics every N epochs
    ckpt_every: int = 5          # save a checkpoint every N epochs
    # EMA momentum is SCHEDULED, not constant: it ramps from ema_start to
    # ema_end over training. The I-JEPA family does this -- early training
    # benefits from a faster-moving target, late training from a
    # near-frozen one. (ema_start matches the official 0.996.)
    ema_start: float = 0.996
    ema_end: float = 1.0


def _ema_momentum(step: int, total: int, start: float, end: float) -> float:
    """EMA decay for the current step: linear ramp from start to end."""
    if total <= 1:
        return end
    frac = min(1.0, step / (total - 1))
    return start + (end - start) * frac


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    """LR multiplier: linear warmup then cosine decay to zero."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: CocktailJEPA, loader: DataLoader, device: str) -> dict:
    """Run the JEPA objective over a loader without training.

    Returns mean loss components plus collapse diagnostics computed on the
    accumulated context embeddings.
    """
    model.eval()
    sums = {"loss": 0.0, "pred_loss": 0.0,
            "sigreg_term": 0.0, "proportion_aux": 0.0}
    n_batches = 0
    emb_chunks = []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        for k in sums:
            sums[k] += float(out[k].item())
        n_batches += 1
        # collect a sample of context embeddings for diagnostics
        if len(emb_chunks) < 16:
            ids, pad = batch["ingredient_ids"], batch["pad_mask"]
            tokens = model.tokens(ids, batch["proportions"])
            ctx = model.context_encoder(
                model.tokens.apply_mask(tokens, batch["mask_index"]), pad)
            emb_chunks.append(ctx[pad].detach().cpu())
    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
    if emb_chunks:
        metrics.update(collapse_report(torch.cat(emb_chunks)))
    model.train()
    return metrics


def _to_device(batch: dict, device: str) -> dict:
    """Move tensor entries of a batch to the device; leave lists alone."""
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train(
    model: CocktailJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    logger,                       # cocktail_jepa.logging.Logger
    run_dir,                      # pathlib.Path for checkpoints
):
    """
    Train the JEPA. Logs loss components and collapse diagnostics every
    epoch. Saves periodic checkpoints and the best-val checkpoint.

    Returns the path of the best checkpoint.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_warmup(s, cfg.warmup_steps, total_steps),
    )

    step = 0
    best_val = float("inf")
    best_path = run_dir / "best.ckpt"
    # tracks the most recent periodic checkpoint so the previous one can
    # be deleted when a newer is written (keep-latest-only, bounds disk)
    prev_periodic = None

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)

            out = model(batch)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            # EMA update of the target encoder AFTER the optimizer step,
            # with a SCHEDULED momentum that ramps over training.
            ema_m = _ema_momentum(step, total_steps,
                                  cfg.ema_start, cfg.ema_end)
            model.ema_update(decay=ema_m)

            epoch_loss += float(loss.item())
            step += 1

        train_loss = epoch_loss / max(1, len(train_loader))
        log_row = {"epoch": epoch, "train_loss": train_loss,
                   "lr": scheduler.get_last_lr()[0],
                   "ema_momentum": _ema_momentum(step, total_steps,
                                                 cfg.ema_start, cfg.ema_end)}

        # validation + collapse diagnostics
        if (epoch + 1) % cfg.val_every == 0:
            val = evaluate(model, val_loader, device)
            log_row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(best_path, model, optimizer, scheduler,
                                step=step, extra={"val_loss": best_val})

        logger.log(log_row, step=epoch)

        # periodic checkpoint -- for crash recovery only.  We keep just
        # the SINGLE most recent one: the older periodic checkpoint is
        # deleted as soon as a newer one is written, so a run's periodic
        # checkpoint footprint stays constant (~one file) instead of
        # growing one file every ckpt_every epochs.  best.ckpt is a
        # separate path and is never touched here.
        if (epoch + 1) % cfg.ckpt_every == 0:
            new_periodic = run_dir / f"epoch_{epoch+1}.ckpt"
            save_checkpoint(new_periodic, model,
                            optimizer, scheduler, step=step)
            if prev_periodic is not None and prev_periodic.exists():
                prev_periodic.unlink()
            prev_periodic = new_periodic

    logger.summary(best_val_loss=best_val, total_steps=step)
    return best_path
