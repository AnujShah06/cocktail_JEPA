"""
diagnostics.py -- collapse diagnostics for JEPA training.

Representational collapse (the encoder mapping every input to the same
vector) is SILENT in the loss curve -- a collapsed model has near-zero
prediction loss and looks like perfect convergence. It must therefore be
detected by inspecting the representations directly.

This module is built BEFORE the training loop, on purpose: the very
first training run must already be instrumented to catch collapse.

Three metrics, computed on a batch of context-encoder embeddings, exactly
as named in the project brief:

  effective_rank   -- how many dimensions the representation genuinely
                      uses, via singular-value entropy. Continuous, so it
                      degrades SMOOTHLY as collapse begins. A healthy
                      model keeps this well above 1; a collapsing model
                      sees it fall toward 1.

  mean_variance    -- average per-dimension variance across the batch.
                      Falls toward 0 as embeddings become constant.

  embedding_spread -- mean pairwise distance between embeddings. Falls
                      toward 0 as all embeddings converge to one point.

NONE of these halt training. They are logged so the operator reads the
W&B curves and judges. (Decided: log-only, no auto-stop.)
"""

from __future__ import annotations

import torch


@torch.no_grad()
def effective_rank(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Effective rank via the entropy of the singular-value distribution.

    embeddings: [N, d] -- a batch of real-slot context embeddings.

    Method: take singular values of the centered embedding matrix,
    normalize them to a probability distribution p_i = s_i / sum(s),
    and compute exp(H) where H = -sum p_i log p_i is the entropy. This
    is the standard "effective rank" (Roy & Vetterli). Ranges from 1
    (all energy in one direction -- collapsed) up to d (energy spread
    evenly -- healthy).
    """
    if embeddings.shape[0] < 2:
        return 1.0
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    # singular values; .float() guards against half-precision SVD issues
    s = torch.linalg.svdvals(x.float())
    s = s[s > eps]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def mean_variance(embeddings: torch.Tensor) -> float:
    """Average per-dimension variance across the batch. -> 0 on collapse."""
    if embeddings.shape[0] < 2:
        return 0.0
    return float(embeddings.var(dim=0).mean().item())


@torch.no_grad()
def embedding_spread(embeddings: torch.Tensor, max_samples: int = 512) -> float:
    """
    Mean pairwise Euclidean distance between embeddings. -> 0 on collapse.

    Subsamples to max_samples rows first so the O(N^2) distance matrix
    stays cheap on large batches.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    if n > max_samples:
        idx = torch.randperm(n, device=embeddings.device)[:max_samples]
        embeddings = embeddings[idx]
    d = torch.cdist(embeddings.float(), embeddings.float())  # [n, n]
    n_eff = d.shape[0]
    # mean of the off-diagonal entries
    total = d.sum() - d.diagonal().sum()
    return float((total / (n_eff * (n_eff - 1))).item())


@torch.no_grad()
def collapse_report(embeddings: torch.Tensor) -> dict[str, float]:
    """Compute all three diagnostics at once. Returns a dict ready to log.

    embeddings: [N, d] real-slot context embeddings from a batch.
    """
    return {
        "effective_rank": effective_rank(embeddings),
        "mean_variance": mean_variance(embeddings),
        "embedding_spread": embedding_spread(embeddings),
    }
