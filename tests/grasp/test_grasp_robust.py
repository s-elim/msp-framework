"""Scoring candidates under a pose posterior and the accept/reject decision (study 4)."""

from __future__ import annotations

import numpy as np

from msp.data.linemod import tcp_pose
from msp.grasp import Reason, RobustScores, decide, score_under_posterior
from msp.pose import exp_so3


def _toy_rollout(T_obj_tcp: np.ndarray, phys: float) -> tuple[float, float, bool]:
    """A 1-D world: the grasp succeeds iff its object-frame x lies in [-w, w], w = phys.
    It is inexecutable when x > 0.05 (jaws on the object)."""
    x = T_obj_tcp[0, 3]
    return float(abs(x) <= phys), 0.0, bool(x <= 0.05)


def test_marginalising_over_pose_prefers_the_robust_candidate():
    # candidate A sits at the edge of the success interval, B in the middle of it
    T = np.stack([tcp_pose(np.array([0.019, 0, 0]), np.array([0, 0, -1.0]), 0.0),
                  tcp_pose(np.array([0.0, 0, 0]), np.array([0, 0, -1.0]), 0.0)])
    phys = [0.02]
    # point estimate (identity pose): both succeed, tie
    point = score_under_posterior(_toy_rollout, T, np.eye(3)[None], np.zeros((1, 3)), phys)
    assert np.array_equal(point.mean, [1.0, 1.0])
    # posterior: the object may sit 5 mm to either side
    ts = np.array([[0.005, 0, 0], [-0.005, 0, 0], [0, 0, 0], [0.004, 0, 0]])
    post = score_under_posterior(_toy_rollout, T, np.repeat(np.eye(3)[None], 4, 0), ts, phys)
    assert post.succ.shape == (4, 1, 2)
    assert post.mean[1] == 1.0 and post.mean[0] < 1.0
    assert decide(post, tau_success=0.5, tau_exec=0.5).index == 1


def test_pose_samples_move_the_object_not_the_gripper():
    """A pose sample translated by +d along camera x puts the TCP at -d in the object frame."""
    seen = []

    def rec(T_obj_tcp: np.ndarray, phys: object) -> tuple[float, float, bool]:
        seen.append(T_obj_tcp[:3, 3].copy())
        return 1.0, 0.0, True

    T = tcp_pose(np.zeros(3), np.array([0, 0, -1.0]), 0.0)[None]
    R = exp_so3(np.array([[0, 0, 0.3]]))
    score_under_posterior(rec, T, R, np.array([[0.01, 0.0, 0.5]]), [None])
    assert np.allclose(seen[0], R[0].T @ (np.zeros(3) - [0.01, 0.0, 0.5]))


def test_decide_reports_the_first_failing_check_and_abstains():
    succ = np.array([[[1.0, 1.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.0]]])  # (M=2, P=1, Na=4)
    exe = np.array([[[True, True, True, False]], [[True, True, True, False]]])
    r = RobustScores(succ=succ.astype(np.float32), exe=exe)
    d = decide(r, tau_success=0.4, tau_exec=0.9, tau_quantile=0.75, q=0.25,
               feasibility=[None, None, None, None])
    assert d.reasons == (Reason.ACCEPTED.value, Reason.NOT_ROBUST.value,
                         Reason.LOW_SUCCESS.value, Reason.NOT_EXECUTABLE.value)
    assert d.index == 0
    d2 = decide(r, tau_success=0.4, tau_exec=0.9, feasibility=["workspace", None, None, None])
    assert d2.reasons[0] == "infeasible:workspace" and d2.index == 1
    none = decide(r, tau_success=1.1)
    assert none.index is None
