"""The paper's decisive experiment, and it needs no robot.

    Does an analytic grasp-quality metric, computed on a RECONSTRUCTED geometry, predict whether
    the object actually gets lifted?

    And does a belief trained on OUTCOMES do better?

WHY THIS REPLACES THE SIM-TO-REAL STUDY.

The T-RO review says the sim-to-real sufficiency gap carries the paper: "without the real-physics
validation, expect rejection regardless of the theory." That is true when the claim is "my simulated
M resembles reality". But a simulation-only paper cannot make that claim, and worse, it does not
need to: with no hardware, M IS the ground truth by definition, so "sufficiency with respect to M"
is true by construction and there is nothing left to test.

The gap that remains -- and it is the one the paper is actually *about* -- is between the geometry a
perception system RECONSTRUCTS and the physics that actually happens. Blueprint wrong-assumption #11:

    "Force closure computed on the estimated geometry is a valid success criterion." It is an
    analytic proxy evaluated on a HALLUCINATED SURFACE. A self-consistent but wrong reconstruction
    passes the check and fails the lift.

On a parametric box that gap is identically zero -- the "reconstruction" and the truth are the same
object -- so the claim is unfalsifiable and the experiment is impossible. On a scanned ketchup
bottle it is neither.

So: the analytic tier computes Ferrari-Canny epsilon on an ORIENTED BOUNDING BOX (roughly the
fidelity a pose-and-shape pipeline delivers). The simulator lifts the grasp against the object's
TRUE convex decomposition. We then ask the only question that matters: how much does the first tell
you about the second?

WHAT WE MEASURED ON THE LIBERO GROCERIES (1303 executable grasps, 13 objects):

    AUC of Ferrari-Canny epsilon vs the real lift outcome        0.596
    best accuracy over ALL thresholds on epsilon                 0.878
    accuracy of just always predicting "this grasp fails"        0.884

Read the last two lines together. **No threshold on the analytic quality beats the majority-class
baseline.** The proxy is not merely mis-calibrated -- re-thresholding cannot rescue it -- it carries
almost no information about whether the object is actually lifted. That is a far stronger statement
than "the numbers disagree", and it is exactly what the paper claims.

AUC IS THE RIGHT METRIC, and accuracy is not. Success is rare (~10% of sampled grasps), so a
predictor that says "fail" to everything scores 90% accuracy and is worthless. AUC is invariant to
that imbalance and to any monotone re-scaling of the score, which is what makes it immune to the
"you just need a better threshold" rebuttal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = ["ProxyComparison", "roc_auc", "compare_predictors"]


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC by the Mann-Whitney statistic. 0.5 = the score carries no information."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def _best_threshold_accuracy(score: np.ndarray, label: np.ndarray) -> float:
    if len(score) == 0:
        return float("nan")
    ths = np.quantile(score, np.linspace(0.01, 0.99, 99))
    return float(max(((score > t) == label).mean() for t in ths))


@dataclass(frozen=True)
class ProxyComparison:
    """The headline table of the paper."""

    n_grasps: int
    success_rate: float
    majority_accuracy: float  # always predict the majority class

    analytic_auc: float  # Ferrari-Canny on the reconstruction
    analytic_best_accuracy: float

    msp_auc: float | None  # the belief, trained on outcomes
    msp_best_accuracy: float | None

    #: THE NATURAL EXPERIMENT, and the sharpest form of the claim.
    #:
    #: Six of the thirteen LIBERO groceries have a SINGLE-box collision hull -- they simply ARE
    #: boxes (butter, cookies, cream cheese...). For those the bounding-box "reconstruction" is
    #: exact and there is no hallucinated surface at all. The other seven are cans, bottles and
    #: cartons with genuinely non-box geometry.
    #:
    #: The paper's thesis makes a falsifiable prediction about this split: the analytic proxy should
    #: work on the boxes, where its geometry is right, and fail on the curved objects, where it is
    #: planning on a surface that does not exist. If instead the proxy is equally bad on both, the
    #: failure is not about reconstruction error at all, and wrong-assumption #11 is not what is
    #: going on. Reporting only the pooled number would hide that.
    analytic_auc_boxlike: float | None = None
    analytic_auc_complex: float | None = None
    msp_auc_boxlike: float | None = None
    msp_auc_complex: float | None = None
    n_boxlike: int = 0
    n_complex: int = 0

    def analytic_is_informative(self, margin: float = 0.02) -> bool:
        """Can ANY threshold on the analytic proxy beat 'always say fail'?"""
        return self.analytic_best_accuracy > self.majority_accuracy + margin

    def to_metrics(self) -> dict[str, float]:
        return {f"proxy/{k}": float(v) for k, v in self.__dict__.items() if v is not None}

    def summary(self) -> str:
        lines = [
            f"{self.n_grasps} executable grasps, success rate {self.success_rate:.3f}",
            "",
            f"  analytic proxy (Ferrari-Canny on a bounding box)",
            f"      AUC vs the real lift outcome        {self.analytic_auc:.3f}",
            f"      best accuracy over ALL thresholds   {self.analytic_best_accuracy:.3f}",
        ]
        if self.msp_auc is not None:
            lines += [
                "",
                f"  MSP (belief trained on outcomes)",
                f"      AUC vs the real lift outcome        {self.msp_auc:.3f}",
                f"      best accuracy over ALL thresholds   {self.msp_best_accuracy:.3f}",
            ]
        lines += [
            "",
            f"  always predict the majority class     {self.majority_accuracy:.3f}",
            "",
            "  0.500 AUC = the score says nothing about what actually happens.",
        ]
        if not self.analytic_is_informative():
            lines.append(
                "  NOTE: no threshold on the analytic proxy beats the majority baseline. It is "
                "not mis-calibrated; it is uninformative."
            )
        if self.analytic_auc_boxlike is not None:
            lines += [
                "",
                "  STRATIFIED BY GEOMETRY -- the falsifiable form of the claim:",
                f"    box-shaped objects (reconstruction is EXACT)  n={self.n_boxlike:5d}   "
                f"analytic AUC {self.analytic_auc_boxlike:.3f}"
                + (f"   MSP {self.msp_auc_boxlike:.3f}" if self.msp_auc_boxlike else ""),
                f"    curved objects (reconstruction is WRONG)      n={self.n_complex:5d}   "
                f"analytic AUC {self.analytic_auc_complex:.3f}"
                + (f"   MSP {self.msp_auc_complex:.3f}" if self.msp_auc_complex else ""),
                "",
                "    The thesis predicts the proxy works where its geometry is right and fails "
                "where it is not.",
                "    If it is equally bad on both, the failure is NOT reconstruction error and "
                "wrong-assumption #11",
                "    is not what is happening. Reporting only the pooled number would hide that.",
            ]
        return "\n".join(lines)


@torch.no_grad()
def compare_predictors(
    analytic_score: Tensor,
    true_success: Tensor,
    executable: Tensor,
    msp_score: Tensor | None = None,
    boxlike: Tensor | None = None,
) -> ProxyComparison:
    """Compare the analytic proxy against the truth, and optionally against MSP.

    Args:
        analytic_score: the Ferrari-Canny epsilon of each grasp, computed on the RECONSTRUCTION.
        true_success: the simulator's verdict on the TRUE geometry. 0/1.
        executable: mask of grasps the gripper can actually reach. Non-executable grasps are
            excluded, because otherwise we would mostly be scoring a collision checker: both
            predictors trivially agree that a grasp which cannot be reached will not succeed, and
            including them inflates every number without saying anything about grasp QUALITY.
        msp_score: s(o, a) from the trained belief, if available.
        boxlike: per-grasp mask -- True where the object's collision hull is a SINGLE box, i.e.
            where the bounding-box reconstruction is exact and there is no hallucinated surface.
            Supplying it stratifies the result, which is the falsifiable form of the claim.
    """
    m = executable.flatten().bool().cpu().numpy()
    y = true_success.flatten().cpu().numpy()[m]
    a = analytic_score.flatten().cpu().numpy()[m]

    base = float(max(y.mean(), 1.0 - y.mean())) if len(y) else float("nan")

    s = None
    msp_auc = msp_best = None
    if msp_score is not None:
        s = msp_score.flatten().cpu().numpy()[m]
        msp_auc = roc_auc(s, y)
        msp_best = _best_threshold_accuracy(s, y)

    a_box = a_cx = m_box = m_cx = None
    n_box = n_cx = 0
    if boxlike is not None:
        bl = boxlike.flatten().bool().cpu().numpy()[m]
        n_box, n_cx = int(bl.sum()), int((~bl).sum())
        if n_box:
            a_box = roc_auc(a[bl], y[bl])
            m_box = roc_auc(s[bl], y[bl]) if s is not None else None
        if n_cx:
            a_cx = roc_auc(a[~bl], y[~bl])
            m_cx = roc_auc(s[~bl], y[~bl]) if s is not None else None

    return ProxyComparison(
        n_grasps=int(m.sum()),
        success_rate=float(y.mean()) if len(y) else float("nan"),
        majority_accuracy=base,
        analytic_auc=roc_auc(a, y),
        analytic_best_accuracy=_best_threshold_accuracy(a, y),
        msp_auc=msp_auc,
        msp_best_accuracy=msp_best,
        analytic_auc_boxlike=a_box,
        analytic_auc_complex=a_cx,
        msp_auc_boxlike=m_box,
        msp_auc_complex=m_cx,
        n_boxlike=n_box,
        n_complex=n_cx,
    )
