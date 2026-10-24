"""The two learned modules (A: encoder, B: outcome head), plus the amortized acquisition net.

Step 5 of the blueprint reduces MSP to exactly two LEARNED MODULES: the encoder and the outcome
head. `AcquisitionNet` is not a third one in the formal sense -- it estimates a functional of those
two (Eq 17) rather than adding a new object to the theory -- but it does carry its own weights, so
it lives here and is trained by its own loss (Eq 18).
"""

from msp.models.nets import (
    AcquisitionNet,
    Backbone,
    BeliefEncoder,
    MLPBackbone,
    OutcomeHead,
    ResNetBackbone,
)

__all__ = [
    "AcquisitionNet",
    "Backbone",
    "BeliefEncoder",
    "MLPBackbone",
    "OutcomeHead",
    "ResNetBackbone",
]
