"""The paper's evidence: identifiability (C2), active perception (C3), and the frontier."""

from msp.diagnostics.active_eval import ActiveReport, evaluate_active_perception
from msp.diagnostics.proxy_eval import ProxyComparison, compare_predictors, roc_auc
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
from msp.diagnostics.selective import (
    within_scene_auc,
    RiskCoverage,
    SelectiveComparison,
    compare_selective,
    risk_coverage,
)

__all__ = [
    "ActiveReport",
    "ProxyComparison",
    "RiskCoverage",
    "SelectiveComparison",
    "compare_predictors",
    "compare_selective",
    "risk_coverage",
    "within_scene_auc",
    "roc_auc",
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
