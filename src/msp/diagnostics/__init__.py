"""The paper's evidence: identifiability (C2), sufficiency, and the rate-distortion frontier."""

from msp.diagnostics.identifiability import (
    IdentifiabilityReport, analyze, measure_invariant_subspace, null_space,
    outcome_jacobian, principal_angles, row_space, subspace_alignment,
)

__all__ = [
    "IdentifiabilityReport", "analyze", "measure_invariant_subspace", "null_space",
    "outcome_jacobian", "principal_angles", "row_space", "subspace_alignment",
]
