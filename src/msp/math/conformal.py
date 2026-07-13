"""Distribution-free calibrated abstention. Contribution C4, Formalization Section 7.

This module carries the framework's only *guarantee*, so it is written defensively and
every failure mode is arranged to fail CLOSED (refuse to act) rather than open (act
without a certificate).

The audited implementation failed open in four independent ways, all reproduced as
regression tests in `tests/math/test_conformal.py`:

  1. It computed only score_1 = 1 - s and never score_0 = s, so it never constructed the
     set C(o,a) at all. Any action whose set was the *ambiguous* {0, 1} was certified --
     a coin-flip action with s = 0.50 passed.
  2. It used `np.quantile`'s default linear interpolation. Split-conformal validity
     requires the ceil((n+1)(1-alpha))-th ORDER STATISTIC. Interpolation returns a smaller
     q_hat and under-covers.
  3. It silently clamped the quantile level to 1.0 when n was too small, voiding the
     finite-sample guarantee without warning.
  4. Its ACI update substituted the *mutating* alpha for the fixed target, turning a stable
     tracking recursion into positive feedback that diverged to the clamp in ~100 steps,
     collapsing a 90% coverage target to 1%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from msp.types import PredictionSet

__all__ = [
    "nonconformity_scores",
    "conformal_quantile",
    "min_calibration_size",
    "prediction_set",
    "AdaptiveConformalState",
]


def nonconformity_scores(s: Tensor, succ: Tensor) -> Tensor:
    """The split-conformal nonconformity score. Formalization Eq 22::

        E(o, a, succ) =  1 - s   if succ = 1
                         s       if succ = 0

    A large score means the model was surprised. Note this is exactly `1 - p_model(y_true)`,
    the standard "one minus the predicted probability of the realized label" score.

    Args:
        s:    (N,) predicted success probabilities s(o, a) from Eq 13.
        succ: (N,) realized binary outcomes.

    Returns:
        (N,) scores in [0, 1].
    """
    if s.shape != succ.shape:
        raise ValueError(f"shape mismatch: s {tuple(s.shape)} vs succ {tuple(succ.shape)}")
    return torch.where(succ > 0.5, 1.0 - s, s)


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration fold for which split conformal is non-vacuous.

    The quantile level is ceil((n+1)(1-alpha)) / n. For this to be <= 1 -- i.e. for the
    required order statistic to exist inside the sample -- we need::

        ceil((n+1)(1-alpha)) <= n     <==>     n >= 1/alpha - 1

    Below this, no finite q_hat can certify anything and the guarantee is vacuous. The
    audited code clamped the level to 1.0 and carried on silently; we raise instead.
    """
    return max(1, math.ceil(1.0 / alpha - 1.0))


def conformal_quantile(scores: Tensor, alpha: float) -> float:
    """q_hat: the ceil((n+1)(1-alpha))-th smallest nonconformity score.
    Formalization Eq 23, and the quantity whose validity Theorem 7 rests on.

    The finite-sample coverage proof needs the empirical quantile taken as an ORDER
    STATISTIC, not by interpolating between neighbouring order statistics. Interpolation
    (numpy's and torch's default) returns a value strictly below the required order
    statistic whenever the level falls between two ranks, which loses coverage. We index the
    sorted scores directly, which is both exactly correct and faster.

    Args:
        scores: (n,) nonconformity scores from a calibration fold that the model was NOT
            fit on (Assumption A5 -- exchangeability). This function cannot verify that;
            `SplitConformalCalibrator` is responsible for enforcing fold disjointness.
        alpha: target miscoverage rate, e.g. 0.1 for 90% coverage.

    Returns:
        q_hat as a float.

    Raises:
        ValueError: if the fold is too small for a non-vacuous guarantee.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    n = scores.numel()
    n_min = min_calibration_size(alpha)
    if n < n_min:
        raise ValueError(
            f"Calibration fold of size n={n} is too small for alpha={alpha}: split conformal "
            f"requires n >= ceil(1/alpha - 1) = {n_min}, otherwise the coverage guarantee is "
            f"vacuous. Collect more calibration data or raise alpha."
        )

    # rank = ceil((n+1)(1-alpha)), 1-indexed -> subtract 1 for 0-indexed gather.
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(rank, n)  # guarded by the n >= n_min check above; belt and braces.
    sorted_scores, _ = torch.sort(scores.flatten())
    return sorted_scores[rank - 1].item()


def prediction_set(s: Tensor, q_hat: float) -> PredictionSet:
    """Construct C(o, a) for every candidate action. Formalization Eq 23::

        C(o, a) = { l in {0,1} : score_l(o, a) <= q_hat },
        score_1 = 1 - s,   score_0 = s

    BOTH memberships are computed. The returned `PredictionSet` exposes
    `is_certified_success()`, which is the singleton test C(o,a) == {1} of Eq 24 -- the
    test the audited code omitted.

    The four possible sets and their meanings:

        {1}     certified success -- eligible for execution
        {0}     certified failure -- the model is confident this grasp fails
        {0,1}   ambiguous        -- both labels plausible. NOT certified. This is the
                                    case the old code wrongly certified.
        {}      empty            -- the model is confidently out of distribution; the
                                    score exceeds q_hat for both labels

    Args:
        s: (B, Na) success probabilities from Eq 13.
        q_hat: the calibrated quantile.

    Returns:
        PredictionSet with boolean fields of shape (B, Na).
    """
    return PredictionSet(
        contains_success=(1.0 - s) <= q_hat,
        contains_failure=s <= q_hat,
    )


@dataclass
class AdaptiveConformalState:
    """Adaptive Conformal Inference (Gibbs & Candes). Formalization Section 7::

        alpha_t <- alpha_t + gamma * (alpha - err_t)

    where `alpha` (here `target`) is FIXED and `err_t` in {0, 1} is the miscoverage
    indicator observed at step t. ACI guarantees long-run coverage under *arbitrary*
    distribution drift -- it does not need exchangeability -- which is what makes the
    certificate survive deployment.

    THE BUG THIS TYPE EXISTS TO PREVENT. The audited code wrote::

        self.alpha = self.alpha + self.gamma * (self.alpha - err_t)

    substituting the mutating `self.alpha` for the fixed target. The recursion becomes
    alpha_{t+1} = alpha_t * (1 + gamma) - gamma * err_t, which is positive feedback with no
    anchor: it diverges to the clamp within ~100 steps, silently turning a 90% coverage
    target into 1%. Here `target` is a separate, immutable field, so the fixed point cannot
    be lost. Regression-tested in `test_aci_is_anchored_to_target`.
    """

    target: float  # the fixed alpha of Theorem 7. NEVER mutated.
    gamma: float = 0.01  # ACI step size
    alpha_t: float = 0.0  # the online, mutating level
    _min: float = 1e-3
    _max: float = 1.0 - 1e-3

    def __post_init__(self) -> None:
        if not 0.0 < self.target < 1.0:
            raise ValueError(f"target alpha must be in (0,1); got {self.target}.")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0; got {self.gamma}.")
        if self.alpha_t == 0.0:
            self.alpha_t = self.target

    def update(self, miscovered: bool) -> float:
        """One ACI step. `miscovered` is err_t: True iff the realized label fell OUTSIDE
        the prediction set at this step.

        Returns the new alpha_t, to be used for the next `conformal_quantile` call.
        """
        err_t = 1.0 if miscovered else 0.0
        self.alpha_t += self.gamma * (self.target - err_t)  # target is FIXED
        self.alpha_t = min(max(self.alpha_t, self._min), self._max)
        return self.alpha_t
