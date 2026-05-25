"""
loss.py -- the JEPA training objective.

Phase-2 fix #18 replaces the old VICReg-style anti-collapse pair (a
variance hinge + a covariance penalty, three tuned weights between them)
with a SINGLE distributional regularizer: SIGReg, from LeJEPA
(Balestriero & LeCun, Nov 2025).

WHY THE CHANGE
--------------
The old objective had two separate anti-collapse terms, each with its own
weight, plus the prediction loss -- a fragile multi-hyperparameter balance
where var/cov weights had to be hand-tuned and could still leave the
representation subtly degenerate.  More importantly, var/cov only
constrain the FIRST TWO MOMENTS of the embedding distribution: the
variance hinge matches per-dim variance, the covariance term decorrelates.
An embedding can satisfy both yet still be very non-Gaussian -- and that
mismatch is exactly the kind of shortcut a JEPA collapses into.

SIGReg instead matches the WHOLE embedding distribution to an isotropic
Gaussian N(0, I) via characteristic-function testing, constraining all
moments at once, with ONE trade-off hyperparameter (sigreg_weight).

WHAT WE KEEP
------------
LeJEPA argues SIGReg makes the EMA teacher-student setup redundant.  We do
NOT take that step: this project's central claim is that the JEPA's latent
prediction error IS an energy function, and the energy estimator
(energy.py) measures prediction error against the EMA target encoder's
latent.  Dropping EMA would remove the thing the energy function is built
on.  So #18 adopts SIGReg purely as a better REGULARIZER, alongside the
existing predictive loss + EMA target encoder -- the same way the LeJEPA
world-model reference (LeWM) uses it.

HOW SIGReg WORKS (Epps-Pulley, the true LeJEPA test)
----------------------------------------------------
A multivariate Gaussianity test is hard directly, but the Cramer-Wold
theorem says: a distribution is multivariate Gaussian iff EVERY 1D
projection of it is univariate Gaussian.  So SIGReg:
  1. projects the embeddings onto many random unit directions -> 1D sets
  2. for each projection, compares its empirical characteristic function
     phi_hat(t) = mean_i exp(i t x_i)  against the characteristic function
     of N(0,1), which is exp(-t^2/2)
  3. integrates the squared discrepancy over a grid of t (the "knots"),
     by the trapezoidal rule, and averages across projections.
The result is 0 when the embeddings are exactly N(0, I) and grows with any
deviation -- collapse, low rank, wrong scale, off-center, or non-Gaussian
shape.  Crucially the projections are NOT standardized before the test:
standardizing would hide exactly the low-variance directions that collapse
and low-rank degeneracy produce.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg_term(
    embeddings: torch.Tensor,
    n_projections: int = 64,
    n_knots: int = 17,
    t_max: float = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularization (Epps-Pulley form).

    embeddings : [N, d]   the set of embeddings to regularize.
    Returns a non-negative scalar: 0 when `embeddings` is distributed as an
    isotropic Gaussian N(0, I), larger the further it deviates (collapse,
    low rank, wrong scale, non-Gaussian shape).

    n_projections : number of random 1D directions (LeJEPA's M). More gives
                    a lower-variance estimate; 64 is ample at d_model=192.
    n_knots       : grid points for the characteristic-function comparison.
    t_max         : the CF is compared on t in [-t_max, t_max]; beyond ~5
                    both CFs are negligible so the integral has converged.

    The random projection directions are resampled every call -- this is
    intended (LeJEPA does likewise): a fresh sketch each step means the
    regularizer is not blind to any fixed subspace.
    """
    n, d = embeddings.shape
    if n < 2:
        return embeddings.new_zeros(())

    device, dtype = embeddings.device, embeddings.dtype

    # random unit-norm projection directions  A : [d, M]
    a = torch.randn(d, n_projections, device=device, dtype=dtype,
                    generator=generator)
    a = a / (a.norm(dim=0, keepdim=True) + 1e-8)

    # project: P : [N, M]  -- RAW projections, deliberately not standardized
    proj = embeddings @ a

    # knots t : [K]
    t = torch.linspace(-t_max, t_max, n_knots, device=device, dtype=dtype)

    # empirical characteristic function per projection:
    #   phi_hat(t) = mean_i exp(i t x_i) = mean_i (cos + i sin)
    # arg : [N, M, K]
    arg = proj.unsqueeze(-1) * t.view(1, 1, -1)
    ecf_re = torch.cos(arg).mean(dim=0)          # [M, K]
    ecf_im = torch.sin(arg).mean(dim=0)          # [M, K]

    # target: CF of N(0,1) is exp(-t^2 / 2), purely real
    tgt_re = torch.exp(-t.pow(2) / 2.0)          # [K]

    # squared discrepancy between empirical and target CF
    disc = (ecf_re - tgt_re.view(1, -1)).pow(2) + ecf_im.pow(2)   # [M, K]

    # trapezoidal integration over t, then average across projections
    integ = torch.trapezoid(disc, t, dim=1)      # [M]
    return integ.mean()


def jepa_loss(
    predicted: torch.Tensor,        # [B, d]  predictor output
    target: torch.Tensor,           # [B, d]  target-encoder latent (detached)
    context_embeddings: torch.Tensor,  # [N, d] real-slot context embeddings
    sigreg_weight: float = 1.0,
    sigreg_projections: int = 64,
) -> dict[str, torch.Tensor]:
    """
    Compute the full JEPA loss: prediction loss + SIGReg regularizer.

    `target` MUST already be detached by the caller (stop-gradient) --
    the target encoder is updated only by EMA, never by this gradient.

    `context_embeddings` is the pooled set of real (non-pad) slot
    embeddings from the context encoder; SIGReg is computed on these so the
    regularizer shapes the representation space itself.

    Returns a dict with the total loss and each component.  The keys
    `var_term` and `cov_term` from the old objective are intentionally
    GONE -- replaced by the single `sigreg_term` -- so any logging or
    diagnostic code that referenced them must be updated (loop.py is).

    sigreg_weight  : LeJEPA's single trade-off hyperparameter (lambda).
                     One number replaces the old var_weight + cov_weight.
    """
    pred_loss = F.smooth_l1_loss(predicted, target)
    sigreg = sigreg_term(context_embeddings, n_projections=sigreg_projections)
    total = pred_loss + sigreg_weight * sigreg
    return {
        "loss": total,
        "pred_loss": pred_loss,
        "sigreg_term": sigreg,
    }
