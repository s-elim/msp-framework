"""Pure mathematics of MSP: one module per section of the formalization.

Nothing in this package imports torch.nn, touches a device, or performs I/O. Every function
corresponds to a numbered equation and is unit-tested against a hand-computed reference, so
the science can be verified without instantiating a network.
"""

from msp.math.bottleneck import (
    BetaSchedule,
    DistortionTerms,
    VIBTerms,
    distortion,
    rate,
    vib_objective,
)
from msp.math.conformal import (
    AdaptiveConformalState,
    conformal_quantile,
    min_calibration_size,
    nonconformity_scores,
    prediction_set,
)
from msp.math.decision import SuccessStats, ambiguity, risk_averse_score, success_stats
from msp.math.divergences import (
    diagonal_gaussian_kl,
    gaussian_nll,
    kl_to_standard_normal,
    log_normal_nll,
    zero_inflated_lognormal_nll,
)

__all__ = [
    "BetaSchedule", "DistortionTerms", "VIBTerms", "distortion", "rate", "vib_objective",
    "AdaptiveConformalState", "conformal_quantile", "min_calibration_size",
    "nonconformity_scores", "prediction_set",
    "SuccessStats", "ambiguity", "risk_averse_score", "success_stats",
    "diagonal_gaussian_kl", "gaussian_nll", "kl_to_standard_normal", "log_normal_nll",
    "zero_inflated_lognormal_nll",
]
