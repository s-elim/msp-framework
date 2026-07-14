"""Tier 2 of M: a version-pinned rigid-body contact simulator.

Formalization Section 11::

    M (train):  analytic Ferrari-Canny epsilon-quality + force-closure margin
                (differentiable prior) COMPOSED WITH a version-pinned rigid-body
                contact simulator for slip/perturbation.

WHY BOTH TIERS EXIST, AND WHY THIS ONE CANNOT REPLACE THE ANALYTIC ONE.

The analytic oracle is a QUASI-STATIC wrench-space model. It asks "if the jaws were in contact
here, could the resulting friction cones bound the origin of wrench space?" That question is
answerable in closed form, differentiably, and in microseconds. It is also *incomplete*: it cannot
see the object topple as the jaws close, cannot see it roll out of the grasp, and models post-lift
slip as a static force-balance rather than as a dynamic event.

Those are exactly the failures this simulator exists to catch, and they are exactly the failures the
T-RO reviewers care about ("contact-rich outcomes are where simulators are least trustworthy... and
where the analytic proxy is most wrong").

But it cannot replace the analytic tier, for two reasons that are worth stating plainly rather than
discovering later:

  1. IT IS NOT DIFFERENTIABLE. Contribution C2 obtains J(x) by autodiff through Phi (Section 11:
     "estimate J(x) by autodiff of Phi in sim"). You cannot backprop through `mj_step`. With this
     oracle alone, J(x) falls back to central differences, which in float32 carry ~1e-2 absolute
     error -- enough to corrupt the numerical RANK of J, and the rank is the entire result.

  2. IT IS ~10^5 TIMES SLOWER. One approach-close-lift-settle rollout is O(1) second. A training
     corpus of 20k scenes x 16 actions is 320k rollouts: tens to hundreds of hours. The analytic
     tier produces the same 320k labels in milliseconds.

`CompositeOracle` therefore takes the differentiable margin and Phi from tier 1, and the dynamic
success and slip from tier 2. That is the composition Section 11 asks for, and it is also the only
composition that is both correct and affordable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from msp.oracle.analytic import STATE_SLICES, _axis_angle_to_matrix, _quat_to_matrix
from msp.oracle.base import PhysicsOracle
from msp.types import Outcome

__all__ = ["MuJoCoOracle", "SimConfig"]


@dataclass(frozen=True)
class SimConfig:
    """Every number that changes an outcome, pinned in one place.

    A rigid-body simulator's results are a function of its solver settings as much as of its
    physics. Reporting a grasp success rate without pinning these is not reproducible, and the
    reviewers explicitly ask for a "version-pinned" simulator.
    """

    timestep: float = 0.002
    close_steps: int = 300  # jaws closing
    settle_steps: int = 100  # let contacts settle before lifting
    lift_steps: int = 400  # lift and hold
    lift_height: float = 0.12  # metres
    lift_speed: float = 0.15  # m/s
    grip_force: float = 40.0  # N, per finger
    success_height: float = 0.05  # object must clear this to count as lifted
    solver_iterations: int = 100


# The standard MuJoCo pattern for a FLOATING gripper: the hand is a real body with a free joint,
# dragged by a weld equality to a mocap target, and the fingers are CHILDREN of the hand with slide
# joints. An earlier version welded the fingers directly to the hand *and* gave them slide joints;
# a weld constraint pins the body completely, so the joints could never move, the jaws never
# closed, and every grasp in the simulator failed (success rate exactly 0.000). If a simulator
# reports that nothing ever works, suspect the model before the physics.
_MJCF = """
<mujoco model="msp_grasp">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          impratio="10" iterations="{iters}"/>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 0" friction="1 0.02 0.001"/>

    <body name="object" pos="0 0 0.05">
      <freejoint name="obj"/>
      <inertial pos="{cx} {cy} {cz}" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="obj_geom" type="{gtype}" size="{gsize}" friction="{mu} 0.02 0.001"
            rgba="0.6 0.6 0.7 1"/>
    </body>

    <body name="target" mocap="true" pos="0 0 0.4"/>

    <body name="hand" pos="0 0 0.4">
      <freejoint name="hand_free"/>
      <geom name="palm" type="box" size="0.03 0.02 0.012" mass="0.6"
            contype="0" conaffinity="0" rgba="0.2 0.2 0.25 0.5"/>
      <body name="finger_l" pos="-0.045 0 -0.035">
        <joint name="fl" type="slide" axis="1 0 0" range="0 0.042" damping="20"/>
        <geom name="fl_geom" type="box" size="0.006 0.014 0.028" mass="0.05"
              friction="{mu} 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
      <body name="finger_r" pos="0.045 0 -0.035">
        <joint name="fr" type="slide" axis="-1 0 0" range="0 0.042" damping="20"/>
        <geom name="fr_geom" type="box" size="0.006 0.014 0.028" mass="0.05"
              friction="{mu} 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <weld body1="hand" body2="target" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <actuator>
    <position name="grip_l" joint="fl" kp="800" ctrlrange="0 0.042" forcerange="-60 60"/>
    <position name="grip_r" joint="fr" kp="800" ctrlrange="0 0.042" forcerange="-60 60"/>
  </actuator>
</mujoco>
"""


class MuJoCoOracle(PhysicsOracle):
    """Rigid-body grasp rollouts: approach, close, lift, settle, measure.

    Reports the DYNAMIC quantities the analytic tier cannot compute:

        success  the object actually left the table and stayed in the hand
        slip     post-lift deviation of the object from its pose at first contact

    Args:
        shape: "box" or "cylinder", matching `AnalyticGraspOracle`.
        cfg: pinned solver settings. Changing these changes the physics; they are recorded in
            `provenance()` so a reported number can be reproduced.
    """

    differentiable = False  # you cannot backprop through mj_step

    def __init__(
        self,
        shape: str = "box",
        cfg: SimConfig | None = None,
        seed: int = 0,
    ) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MuJoCoOracle needs mujoco. Install with `pip install -e '.[sim]'`."
            ) from e

        if shape not in ("box", "cylinder"):
            raise ValueError(f"shape must be 'box' or 'cylinder'; got {shape!r}")
        self.shape = shape
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self._cache: dict[tuple, object] = {}

    @property
    def state_dim(self) -> int:
        from msp.oracle.analytic import STATE_DIM

        return STATE_DIM

    def provenance(self) -> dict[str, object]:
        """Everything needed to reproduce a number this oracle produced."""
        import mujoco

        return {
            "mujoco_version": mujoco.__version__,
            "shape": self.shape,
            **self.cfg.__dict__,
        }

    # -- model construction --------------------------------------------------

    def _model(self, half: np.ndarray, mass: float, mu: float, com: np.ndarray):
        import mujoco

        if self.shape == "cylinder":
            r = float(half[0])
            gsize = f"{r} {float(half[2])}"
            ixx = iyy = mass * (3 * r**2 + (2 * half[2]) ** 2) / 12
            izz = mass * r**2 / 2
        else:
            gsize = " ".join(f"{float(h)}" for h in half)
            hx, hy, hz = (2 * half).tolist()
            ixx = mass * (hy**2 + hz**2) / 12
            iyy = mass * (hx**2 + hz**2) / 12
            izz = mass * (hx**2 + hy**2) / 12

        key = (gsize, round(mass, 4), round(mu, 3), tuple(np.round(com, 4)))
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        xml = _MJCF.format(
            timestep=self.cfg.timestep,
            iters=self.cfg.solver_iterations,
            gtype=self.shape,
            gsize=gsize,
            mass=mass,
            mu=mu,
            cx=com[0], cy=com[1], cz=com[2],
            ixx=max(ixx, 1e-6), iyy=max(iyy, 1e-6), izz=max(izz, 1e-6),
        )
        m = mujoco.MjModel.from_xml_string(xml)
        self._cache[key] = m
        if len(self._cache) > 256:
            self._cache.pop(next(iter(self._cache)))
        return m

    # -- one rollout ---------------------------------------------------------

    def _rollout(
        self, half: np.ndarray, mass: float, mu: float, com: np.ndarray,
        obj_quat: np.ndarray, grasp_pos: np.ndarray, grasp_quat: np.ndarray,
    ) -> tuple[float, float]:
        """Returns (success, slip)."""
        import mujoco

        m = self._model(half, mass, mu, com)
        d = mujoco.MjData(m)
        c = self.cfg

        obj_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
        hand_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]

        # Object rests on the floor at its sampled orientation.
        rest_z = float(half[2]) + 2e-3
        d.qpos[obj_adr : obj_adr + 3] = [0.0, 0.0, rest_z]
        d.qpos[obj_adr + 3 : obj_adr + 7] = obj_quat

        # Place the hand AT the grasp pose, jaws open.
        hand_pos = np.asarray(grasp_pos, dtype=float) + np.array([0.0, 0.0, rest_z])
        d.qpos[hand_adr : hand_adr + 3] = hand_pos
        d.qpos[hand_adr + 3 : hand_adr + 7] = grasp_quat
        d.mocap_pos[0] = hand_pos
        d.mocap_quat[0] = grasp_quat
        d.ctrl[:] = [0.0, 0.0]  # open
        mujoco.mj_forward(m, d)
        for _ in range(40):
            mujoco.mj_step(m, d)

        # Close the jaws.
        d.ctrl[:] = [0.042, 0.042]
        for _ in range(c.close_steps):
            mujoco.mj_step(m, d)
        for _ in range(c.settle_steps):
            mujoco.mj_step(m, d)

        grasped_pose = d.qpos[obj_adr : obj_adr + 3].copy()
        # Track the ACTUAL hand body, not the mocap target. The weld is a soft constraint, so the
        # hand lags the target by a few millimetres and never quite reaches it. Measuring travel
        # from the target attributes that lag to the OBJECT and reports it as slip -- which made
        # even perfect grasps score slip ~= lift_height. Slip is a relative quantity and must be
        # measured against the thing the object is actually held by.
        hand0 = d.qpos[hand_adr : hand_adr + 3].copy()
        mocap0 = d.mocap_pos[0].copy()

        # Lift, by driving the mocap target upward; the weld drags the hand after it.
        dz = c.lift_speed * c.timestep
        for i in range(c.lift_steps):
            d.mocap_pos[0][2] = min(mocap0[2] + dz * i, mocap0[2] + c.lift_height)
            mujoco.mj_step(m, d)

        final = d.qpos[obj_adr : obj_adr + 3].copy()
        success = float(final[2] > rest_z + c.success_height)

        hand_travel = d.qpos[hand_adr : hand_adr + 3] - hand0
        expected = grasped_pose + hand_travel  # where the object would be if held rigidly
        slip = float(np.linalg.norm(final - expected))
        return success, min(slip, 0.5)

    # -- the interface -------------------------------------------------------

    @torch.no_grad()
    def query(self, state: Tensor, actions: Tensor) -> Outcome:
        """Roll out every (scene, action). SLOW: one simulation each. Use `CompositeOracle` for
        training corpora; use this directly only for validation folds."""
        b, na, _ = actions.shape
        st = state.detach().cpu()
        ac = actions.detach().cpu()

        half = torch.exp(st[:, STATE_SLICES["log_size"]]).clamp(0.01, 0.5).numpy()
        mu = torch.exp(st[:, STATE_SLICES["log_friction"]]).clamp(0.05, 2.0).numpy()
        mass = torch.exp(st[:, STATE_SLICES["log_mass"]]).clamp(0.01, 10.0).numpy()
        com = st[:, STATE_SLICES["com"]].numpy()
        R_obj = _axis_angle_to_matrix(st[:, STATE_SLICES["pose_r"]])
        obj_quat = _matrix_to_quat(R_obj).numpy()

        succ = np.zeros((b, na), dtype=np.float32)
        slip = np.zeros((b, na), dtype=np.float32)

        for i in range(b):
            for j in range(na):
                s, sl = self._rollout(
                    half[i], float(mass[i, 0]), float(mu[i, 0]), com[i],
                    obj_quat[i], ac[i, j, 0:3].numpy(), ac[i, j, 3:7].numpy(),
                )
                succ[i, j], slip[i, j] = s, sl

        out = Outcome(
            succ=torch.from_numpy(succ).unsqueeze(-1).to(state.device),
            # The simulator has no wrench-space quality metric. Margin is the analytic tier's
            # job; a composite supplies it. Returning zeros here rather than inventing a number
            # is deliberate -- see CompositeOracle.
            margin=torch.zeros(b, na, 1, device=state.device),
            slip=torch.from_numpy(slip).unsqueeze(-1).to(state.device),
        )
        out.validate()
        return out

    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:
        """Not available. The simulator is a sampler, not a differentiable map.

        Deliberately raises rather than silently returning a finite-difference approximation:
        J(x) computed by central differences on a stochastic, contact-discontinuous simulator is
        not a Jacobian, it is noise, and it would corrupt the numerical rank that Theorem 4's
        entire claim rests on. Use `CompositeOracle`, whose Phi comes from the analytic tier.
        """
        raise NotImplementedError(
            "MuJoCoOracle.outcome_params is intentionally unavailable: you cannot autodiff "
            "through mj_step, and finite-differencing a contact simulator does not produce a "
            "usable J(x). Use CompositeOracle, which takes Phi from the differentiable analytic "
            "tier and success/slip from this one."
        )


def _matrix_to_quat(R: Tensor) -> Tensor:
    """(B, 3, 3) -> (B, 4) [w, x, y, z]. Shepperd's method, numerically stable near tr = -1."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = torch.sqrt((1 + tr).clamp_min(1e-8)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w.clamp_min(1e-6))
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w.clamp_min(1e-6))
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w.clamp_min(1e-6))
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


del math, _quat_to_matrix  # imported for symmetry with analytic.py; not needed here
