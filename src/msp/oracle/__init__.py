"""Outcome oracles. `LinemodGraspSim` (MuJoCo rollouts on the LINEMOD meshes) labels
LM-Grasp; `SyntheticOracle` is the closed-form world behind `msp.data.synthetic`, used by the
unit tests."""

from msp.oracle.base import PhysicsOracle
from msp.oracle.synthetic import SyntheticOracle

__all__ = ["PhysicsOracle", "SyntheticOracle"]
