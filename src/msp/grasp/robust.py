"""Grasp candidates scored under a pose posterior, and the accept/reject decision.

A pose-then-plan pipeline simulates each candidate once per physics draw at the ESTIMATED pose.
Here each candidate is simulated at M pose samples x P physics draws. The same P draws are used
for every pose sample and every candidate (common random numbers), so differences between
candidates and between pose samples are not physics-sampling noise.

Shapes: Na candidates, M pose samples, P physics draws. Candidate grasps are TCP poses in the
CAMERA frame (Na, 4, 4); pose samples map model points into the camera frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = ["GraspDecision", "Reason", "RobustScores", "decide", "decide_summary",
           "score_under_posterior"]

#: (T_obj_tcp (4, 4), physics parameters) -> (success in {0, 1}, slip m, executable)
Rollout = Callable[[np.ndarray, Any], tuple[float, float, bool]]


@dataclass(frozen=True)
class RobustScores:
    succ: np.ndarray  # (M, P, Na) float32 in {0, 1}
    exe: np.ndarray  # (M, P, Na) bool: open jaws clear of object and table at the grasp pose

    def __post_init__(self) -> None:
        assert self.succ.ndim == 3 and self.succ.shape == self.exe.shape

    @property
    def mean(self) -> np.ndarray:
        """(Na,) posterior-mean success: the ranking score."""
        return self.succ.mean((0, 1))

    @property
    def p_exec(self) -> np.ndarray:
        """(Na,) probability over the posterior that the grasp pose is collision-free."""
        return self.exe.mean((0, 1))

    @property
    def per_pose(self) -> np.ndarray:
        """(M, Na) success averaged over physics, per pose sample."""
        return self.succ.mean(1)

    def quantile(self, q: float) -> np.ndarray:
        """(Na,) q-quantile over pose samples of the physics-averaged success."""
        return np.asarray(np.quantile(self.per_pose, q, axis=0))


def score_under_posterior(rollout: Rollout, T_cam_tcp: np.ndarray, R: np.ndarray,
                          t: np.ndarray, phys: Sequence[Any]) -> RobustScores:
    """Roll out every candidate at every (pose sample, physics draw).

    T_cam_tcp (Na, 4, 4); R (M, 3, 3) and t (M, 3) pose samples (model -> camera, metres);
    phys: P physics parameter sets passed through to `rollout`.
    """
    assert T_cam_tcp.ndim == 3 and T_cam_tcp.shape[1:] == (4, 4), T_cam_tcp.shape
    assert R.ndim == 3 and t.shape == (R.shape[0], 3), (R.shape, t.shape)
    m, p, na = R.shape[0], len(phys), T_cam_tcp.shape[0]
    succ = np.zeros((m, p, na), np.float32)
    exe = np.zeros((m, p, na), bool)
    T_co = np.zeros((m, 4, 4))
    T_co[:, :3, :3], T_co[:, :3, 3], T_co[:, 3, 3] = R, t, 1.0
    T_obj = np.linalg.inv(T_co)[:, None] @ T_cam_tcp[None]  # (M, Na, 4, 4)
    for b, ph in enumerate(phys):  # physics outermost: the simulator caches the rest pose
        for a in range(m):
            for j in range(na):
                s, _, e = rollout(T_obj[a, j], ph)
                succ[a, b, j], exe[a, b, j] = s, e
    return RobustScores(succ=succ, exe=exe)


class Reason(str, Enum):
    ACCEPTED = "accepted"
    NOT_EXECUTABLE = "collision_prob"  # P(open jaws hit object or table) too high
    LOW_SUCCESS = "low_expected_success"
    NOT_ROBUST = "low_success_quantile"  # fails under a plausible share of the pose posterior
    INFEASIBLE = "infeasible"  # frame / workspace / kinematic check (see msp.grasp.frames)


@dataclass(frozen=True)
class GraspDecision:
    index: int | None  # chosen candidate, None = abstain
    reasons: tuple[str, ...]  # per candidate
    score: np.ndarray  # (Na,) posterior mean


def decide(scores: RobustScores, tau_success: float = 0.5, tau_exec: float = 0.9,
           tau_quantile: float | None = None, q: float = 0.25,
           feasibility: Sequence[str | None] | None = None) -> GraspDecision:
    """Highest posterior-mean candidate among those passing every check, else abstain.

    feasibility: per candidate None (feasible) or the failed check's name, from
    `msp.grasp.frames.check_grasp`. The first failing reason is reported per candidate, in the
    order infeasible, collision, low success, low quantile.
    """
    qv = scores.quantile(q) if tau_quantile is not None else None
    return decide_summary(scores.mean, scores.p_exec, qv, tau_success, tau_exec, tau_quantile,
                          feasibility)


def decide_summary(mean: np.ndarray, pe: np.ndarray, qv: np.ndarray | None = None,
                   tau_success: float = 0.5, tau_exec: float = 0.9,
                   tau_quantile: float | None = None,
                   feasibility: Sequence[str | None] | None = None) -> GraspDecision:
    """`decide` on per-candidate summaries (Na,): posterior mean, executable probability and
    the pose quantile, e.g. as stored in a plan file."""
    na = len(mean)
    feas = list(feasibility) if feasibility is not None else [None] * na
    assert len(feas) == na
    reasons = []
    for j in range(na):
        if feas[j] is not None:
            reasons.append(f"{Reason.INFEASIBLE.value}:{feas[j]}")
        elif pe[j] < tau_exec:
            reasons.append(Reason.NOT_EXECUTABLE.value)
        elif mean[j] < tau_success:
            reasons.append(Reason.LOW_SUCCESS.value)
        elif qv is not None and tau_quantile is not None and qv[j] < tau_quantile:
            reasons.append(Reason.NOT_ROBUST.value)
        else:
            reasons.append(Reason.ACCEPTED.value)
    ok = np.array([r == Reason.ACCEPTED.value for r in reasons])
    idx = int(np.argmax(np.where(ok, mean, -np.inf))) if ok.any() else None
    return GraspDecision(index=idx, reasons=tuple(reasons), score=mean)
