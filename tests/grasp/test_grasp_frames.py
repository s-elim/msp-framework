"""Frame chain camera -> base -> flange and the pre-execution checks."""

from __future__ import annotations

import numpy as np
import pytest

from msp.data.linemod import tcp_pose
from msp.grasp import Workspace, check_grasp, grasp_in_base, plan_execution, pregrasp
from msp.pose import exp_so3, make_T

# camera 0.6 m above the table, looking straight down (camera z = base -z), base x = camera x
T_BASE_CAM = make_T(np.diag([1.0, -1.0, -1.0]), np.array([0.3, 0.0, 0.6]))


def test_chain_composes_and_pregrasp_backs_off_along_the_approach():
    # top-down grasp 0.55 m in front of the camera: approach = camera +z = base -z
    T_cam_tcp = tcp_pose(np.array([0.02, 0.01, 0.55]), np.array([0, 0, 1.0]), 0.4)
    T = grasp_in_base(T_BASE_CAM, T_cam_tcp)
    assert np.allclose(T[:3, 3], [0.32, -0.01, 0.05])
    assert np.allclose(-T[:3, 2], [0, 0, -1.0])  # approach straight down in the base frame
    pre = pregrasp(T, 0.1)
    assert np.allclose(pre[:3, 3], T[:3, 3] + [0, 0, 0.1])
    assert np.allclose(pre[:3, :3], T[:3, :3])


def test_checks_name_the_failure():
    ws = Workspace(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.5), max_tilt_deg=45.0, table_z=0.0,
                   min_tcp_height=0.01, standoff=0.1)
    down = tcp_pose(np.array([0.3, 0.0, 0.05]), np.array([0, 0, -1.0]), 0.0)
    assert check_grasp(down, ws) is None
    assert check_grasp(tcp_pose(np.array([0.8, 0, 0.05]), np.array([0, 0, -1.0]), 0), ws) \
        == "workspace"
    side = tcp_pose(np.array([0.3, 0.0, 0.05]), np.array([1.0, 0, -0.2]), 0.0)
    assert check_grasp(side, ws) == "approach_tilt"
    assert check_grasp(tcp_pose(np.array([0.3, 0, 0.005]), np.array([0, 0, -1.0]), 0), ws) \
        == "table_clearance"
    assert check_grasp(tcp_pose(np.array([0.3, 0, 0.45]), np.array([0, 0, -1.0]), 0), ws) \
        == "pregrasp_workspace"
    assert check_grasp(down, ws, reachable=lambda T: T[2, 3] < 0.1) == "ik_pregrasp"


def test_plan_execution_applies_the_flange_offset_and_never_claims_ik():
    T_tcp_flange = make_T(np.eye(3), np.array([0.0, 0.0, 0.12]))  # flange 12 cm behind the TCP
    T_cam_tcp = tcp_pose(np.array([0.0, 0.0, 0.5]), np.array([0, 0, 1.0]), 0.0)
    p = plan_execution(T_BASE_CAM, T_cam_tcp, Workspace(), T_tcp_flange)
    assert p.ok and not p.kinematics_checked
    assert np.allclose(p.T_base_flange[:3, 3], p.T_base_tcp[:3, 3] + [0, 0, 0.12])
    assert set(p.to_dict()) >= {"ok", "T_base_flange", "T_base_flange_pre"}


def test_invalid_transforms_are_rejected():
    bad = np.eye(4)
    bad[:3, :3] = exp_so3(np.array([0.1, 0.2, 0.3])) * 1.01
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        grasp_in_base(bad, np.eye(4))
    with pytest.raises(ValueError, match="4, 4"):
        grasp_in_base(np.eye(3), np.eye(4))
