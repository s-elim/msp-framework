"""The paper's evidence: identifiability (C2), active perception (C3), and the frontier."""

from msp.diagnostics.active_eval import ActiveReport, evaluate_active_perception
from msp.diagnostics.identifiability import (
    IdentifiabilityReport,
    analyze,
    measure_invariant_subspace,
    null_space,
    outcome_jacobian,
    principal_angles,
    row_space,
    subspace_alignment,
)

__all__ = [
    "ActiveReport",
    "IdentifiabilityReport",
    "analyze",
    "evaluate_active_perception",
    "measure_invariant_subspace",
    "null_space",
    "outcome_jacobian",
    "principal_angles",
    "row_space",
    "subspace_alignment",
]
