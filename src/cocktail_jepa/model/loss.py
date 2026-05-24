"""
loss.py -- the JEPA training objective.

Three terms, summed:
  1. prediction loss -- smooth-L1 between the predicted latent and the
     (stop-gradient) target latent. This is the core JEPA objective.
  2. variance term   -- penalizes any embedding dimension whose std across
     the batch falls below 1.0. Forbids the constant-vector collapse.
  3. covariance term -- penalizes off-diagonal correlations between
     embedding dimensions. Forbids dimensions becoming redundant copies.

Terms 2 and 3 are the VICReg-style anti-collapse regularizers. Collapse
does not show up in the prediction loss -- a model that maps everything
to a constant has ZERO prediction loss -- so these terms are what make
the objective safe. They are built in from Stage 2; Stage 3 only tunes
their weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_term(embeddings: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge penalty: each dimension's batch std should be >= 1.

    embeddings: [N, d]. Returns a scalar. Zero when every dimension is
    sufficiently spread; positive (and gradient-bearing) when a dimension
    is collapsing toward constant.
    """
    std = torch.sqrt(embeddings.var(dim=0) + eps)        # [d]
    return torch.mean(F.relu(1.0 - std))


def covariance_term(embeddings: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance between embedding dimensions.

    embeddings: [N, d]. Returns a scalar. Drives dimensions to be
    decorrelated, so they carry independent information rather than
    redundant copies of the same feature.
    """
    n, d = embeddings.shape
    if n < 2:
        return embeddings.new_zeros(())
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n - 1)              # [d, d]
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def jepa_loss(
    predicted: torch.Tensor,        # [B, d]  predictor output
    target: torch.Tensor,           # [B, d]  target-encoder latent (detached)
    context_embeddings: torch.Tensor,  # [N, d] real-slot context embeddings
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
) -> dict[str, torch.Tensor]:
    """
    Compute the full JEPA loss.

    `target` MUST already be detached by the caller (stop-gradient) --
    the target encoder is updated only by EMA, never by this gradient.

    `context_embeddings` is the pooled set of real (non-pad) slot
    embeddings from the context encoder; the var/cov terms are computed
    on these so the regularizer shapes the representation space itself.

    Returns a dict with the total loss and each component, so Stage 3 can
    log them separately (essential for diagnosing collapse).
    """
    pred_loss = F.smooth_l1_loss(predicted, target)
    var = variance_term(context_embeddings)
    cov = covariance_term(context_embeddings)
    total = pred_loss + var_weight * var + cov_weight * cov
    return {
        "loss": total,
        "pred_loss": pred_loss,
        "var_term": var,
        "cov_term": cov,
    }
