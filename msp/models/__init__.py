"""Models module initialization."""

from msp.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
ENCODER_REGISTRY = Registry("ENCODER")
HEAD_REGISTRY = Registry("HEAD")
READOUT_REGISTRY = Registry("READOUT")

__all__ = [
    "BACKBONE_REGISTRY",
    "ENCODER_REGISTRY",
    "HEAD_REGISTRY",
    "READOUT_REGISTRY"
]
