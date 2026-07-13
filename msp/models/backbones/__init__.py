"""Backbones module initialization."""

# Import base first
from .base import BaseBackbone

# Import vision to trigger registry decorator execution
from .vision import ResNetBackbone

__all__ = ["BaseBackbone", "ResNetBackbone"]
