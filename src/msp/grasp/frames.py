"""Frame chain from a camera-frame grasp to a robot command, and pre-execution checks.

Frames (all 4x4 homogeneous, metres; T_a_b maps coordinates in frame b into frame a)
    base     robot base; +z up, opposite to gravity
    cam      camera optical frame (x right, y down, z forward), as in BOP / LINEMOD
    obj      CAD model frame of the object
    tcp      tool centre point between the finger pads; local -z is the approach direction and
             local x the closing axis (`msp.data.linemod.tcp_pose`)
    flange   the arm's tool flange, the frame an IK solver or Cartesian controller takes

    T_base_cam   hand-eye calibration (eye-to-hand), or forward kinematics x T_ee_cam
                 (eye-in-hand)
    T_tcp_flange gripper mounting: flange pose expressed in the TCP frame

Nothing here sends a command. `plan_execution` returns the targets and the result of every
check; an executor must refuse a plan whose `ok` is False.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["ExecutionPlan", "Workspace", "check_grasp", "grasp_in_base", "plan_execution",
           "pregrasp", "validate_transform"]


def validate_transform(T: np.ndarray, name: str = "T", tol: float = 1e-6) -> np.ndarray:
    """Return T as float64 after checking it is a proper rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: expected (4, 4), got {T.shape}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol) or not np.isclose(np.linalg.det(R), 1.0,
                                                                         atol=tol):
        raise ValueError(f"{name}: rotation block is not in SO(3)")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name}: last row must be [0, 0, 0, 1]")
    if not np.isfinite(T).all():
        raise ValueError(f"{name}: non-finite entries")
    return T


def grasp_in_base(T_base_cam: np.ndarray, T_cam_tcp: np.ndarray) -> np.ndarray:
    T: np.ndarray = validate_transform(T_base_cam, "T_base_cam") @ validate_transform(
        T_cam_tcp, "T_cam_tcp")
    return T


def pregrasp(T_base_tcp: np.ndarray, standoff: float) -> np.ndarray:
    """TCP pose `standoff` metres back along the approach (the TCP's local +z)."""
    off = np.eye(4)
    off[2, 3] = standoff
    return T_base_tcp @ off


@dataclass(frozen=True)
class Workspace:
    """Conservative limits in the BASE frame; defaults are placeholders, set them per robot."""

    lo: tuple[float, float, float] = (-0.6, -0.6, 0.0)
    hi: tuple[float, float, float] = (0.6, 0.6, 0.6)
    max_tilt_deg: float = 60.0  # approach direction vs straight down (-z of base)
    table_z: float = 0.0  # support surface height in the base frame
    min_tcp_height: float = 0.01  # TCP above the support surface
    standoff: float = 0.10  # pre-grasp distance along the approach


def check_grasp(T_base_tcp: np.ndarray, ws: Workspace,
                reachable: Callable[[np.ndarray], bool] | None = None,
                T_tcp_flange: np.ndarray | None = None) -> str | None:
    """None if the grasp passes every check, else the name of the first failed check.

    `reachable` is the robot's IK feasibility test on a FLANGE pose in the base frame; without
    it, kinematic feasibility is NOT checked and the caller must say so.
    """
    T = validate_transform(T_base_tcp, "T_base_tcp")
    p = T[:3, 3]
    if np.any(p < np.asarray(ws.lo)) or np.any(p > np.asarray(ws.hi)):
        return "workspace"
    approach = -T[:3, 2]
    tilt = np.degrees(np.arccos(np.clip(approach @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0)))
    if tilt > ws.max_tilt_deg:
        return "approach_tilt"
    if p[2] - ws.table_z < ws.min_tcp_height:
        return "table_clearance"
    pre = pregrasp(T, ws.standoff)
    if np.any(pre[:3, 3] < np.asarray(ws.lo)) or np.any(pre[:3, 3] > np.asarray(ws.hi)):
        return "pregrasp_workspace"
    if reachable is not None:
        Tf = T_tcp_flange if T_tcp_flange is not None else np.eye(4)
        for name, X in (("ik_pregrasp", pre), ("ik_grasp", T)):
            if not reachable(X @ Tf):
                return name
    return None


@dataclass(frozen=True)
class ExecutionPlan:
    ok: bool
    reason: str | None
    T_base_tcp: np.ndarray  # (4, 4) grasp
    T_base_pregrasp: np.ndarray  # (4, 4)
    T_base_flange: np.ndarray  # (4, 4) flange target at the grasp
    T_base_flange_pre: np.ndarray  # (4, 4) flange target at the pre-grasp
    kinematics_checked: bool

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "reason": self.reason,
                "kinematics_checked": self.kinematics_checked,
                "T_base_tcp": self.T_base_tcp.tolist(),
                "T_base_pregrasp": self.T_base_pregrasp.tolist(),
                "T_base_flange": self.T_base_flange.tolist(),
                "T_base_flange_pre": self.T_base_flange_pre.tolist()}


def plan_execution(T_base_cam: np.ndarray, T_cam_tcp: np.ndarray, ws: Workspace,
                   T_tcp_flange: np.ndarray | None = None,
                   reachable: Callable[[np.ndarray], bool] | None = None) -> ExecutionPlan:
    """Targets for one selected grasp plus the result of `check_grasp`; sends nothing."""
    Tf = validate_transform(T_tcp_flange if T_tcp_flange is not None else np.eye(4),
                            "T_tcp_flange")
    T = grasp_in_base(T_base_cam, T_cam_tcp)
    pre = pregrasp(T, ws.standoff)
    reason = check_grasp(T, ws, reachable, Tf)
    return ExecutionPlan(ok=reason is None, reason=reason, T_base_tcp=T, T_base_pregrasp=pre,
                         T_base_flange=T @ Tf, T_base_flange_pre=pre @ Tf,
                         kinematics_checked=reachable is not None)
