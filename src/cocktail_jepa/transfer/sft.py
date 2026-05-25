"""
sft.py -- the supervised-fine-tuning transfer experiment.

This is the brief's proof that the JEPA learned a transferable
representation. The experiment:

  1. Take the pretrained context encoder, FROZEN. Attach a small
     classification head. Train ONLY the head to predict a recipe's
     base-spirit family -- with the base-spirit slot MASKED, so the task
     is genuine inference, not reading the spirit off the input.
  2. Do the same with a FROM-SCRATCH encoder (same architecture, random
     weights, also frozen) + its own head.
  3. Compare accuracy, especially in the LOW-LABEL regime (train the head
     on only a small slice of the labelled data).

If the pretrained encoder beats from-scratch -- particularly when labels
are scarce -- the self-supervised JEPA representation genuinely captured
cocktail structure. That gap is the transfer result.

The JEPA itself is never updated here; only the small head trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocktail_jepa.model.jepa import CocktailJEPA
from cocktail_jepa.transfer.labels import SPIRIT_CLASSES


class SpiritHead(nn.Module):
    """A small classification head: pooled recipe embedding -> spirit class."""

    def __init__(self, d_model: int, n_classes: int = len(SPIRIT_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


@torch.no_grad()
def encode_recipes(
    model: CocktailJEPA,
    batch: dict,
    device: str,
) -> torch.Tensor:
    """
    Produce one pooled embedding per recipe, with the base-spirit slot
    MASKED so the head cannot trivially read the spirit off the input.

    `batch` carries (besides the usual tensors) `spirit_index` [B] -- the
    slot holding the base spirit. That slot is replaced with [MASK]; the
    encoder runs; the real-slot embeddings are mean-pooled.
    """
    ids = batch["ingredient_ids"].to(device)
    props = batch["proportions"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    spirit_index = batch["spirit_index"].to(device)

    tokens = model.tokens(ids, props)
    # mask the base-spirit slot -- the task must infer it, not read it
    tokens = model.tokens.apply_mask(tokens, spirit_index)
    emb = model.context_encoder(tokens, pad_mask)          # [B, L, d]

    # mean-pool over real (non-pad) slots
    mask = pad_mask.unsqueeze(-1).float()
    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


def train_head(
    model: CocktailJEPA,
    head: SpiritHead,
    train_batches: list[dict],
    val_batches: list[dict],
    device: str,
    epochs: int = 30,
    lr: float = 1e-3,
) -> dict:
    """
    Train ONLY the head; the encoder is frozen. Returns best val accuracy.
    """
    model.to(device).eval()
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    best_acc = 0.0
    for _ in range(epochs):
        head.train()
        for batch in train_batches:
            pooled = encode_recipes(model, batch, device)   # no grad to enc
            logits = head(pooled)
            loss = F.cross_entropy(logits, batch["label"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate_head(model, head, val_batches, device)
        best_acc = max(best_acc, acc)
    return {"best_val_acc": best_acc}


@torch.no_grad()
def evaluate_head(
    model: CocktailJEPA,
    head: SpiritHead,
    batches: list[dict],
    device: str,
) -> float:
    """Classification accuracy of the head over a set of batches."""
    head.eval()
    correct = total = 0
    for batch in batches:
        pooled = encode_recipes(model, batch, device)
        pred = head(pooled).argmax(dim=1)
        labels = batch["label"].to(device)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
    return correct / max(1, total)


def run_transfer_comparison(
    pretrained: CocktailJEPA,
    from_scratch: CocktailJEPA,
    train_batches: list[dict],
    val_batches: list[dict],
    device: str,
    label_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    epochs: int = 30,
) -> dict:
    """
    The headline transfer experiment.

    For each label fraction, train a fresh head on that slice of the
    training data -- once on the PRETRAINED frozen encoder, once on the
    FROM-SCRATCH frozen encoder -- and record val accuracy. The pretrained
    advantage should be largest when labels are scarce.

    Returns {fraction: {"pretrained": acc, "from_scratch": acc}}.
    """
    d_model = pretrained.cfg.d_model
    n_train = len(train_batches)
    results: dict = {}

    for frac in label_fractions:
        k = max(1, int(n_train * frac))
        slice_batches = train_batches[:k]

        pre_head = SpiritHead(d_model)
        pre = train_head(pretrained, pre_head, slice_batches, val_batches,
                         device, epochs=epochs)

        scr_head = SpiritHead(d_model)
        scr = train_head(from_scratch, scr_head, slice_batches, val_batches,
                         device, epochs=epochs)

        results[frac] = {
            "pretrained": pre["best_val_acc"],
            "from_scratch": scr["best_val_acc"],
            "n_train_batches": k,
        }
    return results


def format_transfer_report(results: dict, n_classes: int) -> str:
    """Pretty-print the transfer comparison."""
    chance = 1.0 / n_classes
    lines = [
        "SFT TRANSFER EXPERIMENT",
        "=" * 52,
        "",
        f"Task: classify base-spirit family ({n_classes} classes, "
        f"chance = {chance:.2f})",
        "Encoder is FROZEN; only the classification head is trained.",
        "The base-spirit slot is masked -- the head must INFER the family.",
        "",
        f"{'label fraction':>16} | {'pretrained':>11} | "
        f"{'from-scratch':>12} | {'gap':>6}",
        "-" * 52,
    ]
    for frac in sorted(results):
        r = results[frac]
        gap = r["pretrained"] - r["from_scratch"]
        lines.append(
            f"{frac:>15.0%}  | {r['pretrained']:>10.3f}  | "
            f"{r['from_scratch']:>11.3f}  | {gap:>+6.3f}"
        )
    lines += [
        "",
        "Interpretation: a positive gap means the self-supervised JEPA",
        "representation transfers. The gap should be LARGEST at small",
        "label fractions -- that is the value of pretraining.",
    ]
    return "\n".join(lines)
