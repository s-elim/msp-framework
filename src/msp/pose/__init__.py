"""Pose uncertainty for model-based pose estimators that report a point estimate only."""

from msp.pose.features import FEATURE_NAMES, ModelSurface, pose_features
from msp.pose.residuals import (
    PosteriorSampler,
    ResidualBank,
    apply_residuals,
    coverage,
    fit_bank,
    pose_residual,
)
from msp.pose.se3 import (
    exp_so3,
    log_so3,
    make_T,
    random_rotation,
    reduce_symmetry,
    rotation_angle,
    symmetry_transforms,
)

__all__ = [
    "FEATURE_NAMES",
    "ModelSurface",
    "PosteriorSampler",
    "ResidualBank",
    "apply_residuals",
    "coverage",
    "exp_so3",
    "fit_bank",
    "log_so3",
    "make_T",
    "pose_features",
    "pose_residual",
    "random_rotation",
    "reduce_symmetry",
    "rotation_angle",
    "symmetry_transforms",
]
