"""Physics module initialization."""

from msp.utils.registry import Registry

ORACLE_REGISTRY = Registry("ORACLE")
SAMPLER_REGISTRY = Registry("SAMPLER")

__all__ = ["ORACLE_REGISTRY", "SAMPLER_REGISTRY"]
