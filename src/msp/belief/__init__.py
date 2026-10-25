"""Posterior families for the image code z of the learned outcome scorers."""

from msp.belief.base import Belief
from msp.belief.gaussian import DiagonalGaussianBelief

__all__ = ["Belief", "DiagonalGaussianBelief"]
