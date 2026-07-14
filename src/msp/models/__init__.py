"""The two learned modules: A (BeliefEncoder) and B (OutcomeHead).

Step 5 of the blueprint reduces MSP to exactly these two. Selection, abstention, active
perception and adaptation are inference procedures over them, not further networks.

An `AcquisitionNet` for Eq 18 lived here briefly and has been REMOVED. Eq 18 regresses onto
IG_true, which requires a rendered look-ahead (Eq 17) that does not exist yet. A network with
no trainable target is not a feature -- the audited repo shipped exactly that and then took an
argmax over its random weights to steer a camera. It comes back when the renderer does.
"""

from msp.models.nets import (
    Backbone, BeliefEncoder, MLPBackbone, OutcomeHead, ResNetBackbone,
)

__all__ = ["Backbone", "BeliefEncoder", "MLPBackbone", "OutcomeHead", "ResNetBackbone"]
