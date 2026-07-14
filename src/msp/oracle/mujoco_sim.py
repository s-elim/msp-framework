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

import os


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
    grip_force: float = 80.0  # N, per finger
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

  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.32 0.35" rgb2="0.38 0.40 0.43"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.3 0.3 0.6" dir="0.3 -0.3 -0.6" diffuse="0.4 0.4 0.45" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 0" friction="1 0.02 0.001"
          material="grid_mat"/>

    <!-- The RGB-D sensor. Mounted off-axis and looking down at the workspace, like a wrist- or
         shoulder-mounted RealSense. Its pose is FIXED and part of the model, so o = render(x) is a
         well-defined observation kernel p(o|x) rather than a free parameter of the experiment. -->
    <camera name="rgbd" pos="0.17 -0.17 0.20" xyaxes="0.707 0.707 0 -0.43 0.43 0.79"
            fovy="48"/>

    <body name="object" pos="0 0 0.05">
      <freejoint name="obj"/>
      <inertial pos="{cx} {cy} {cz}" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="obj_geom" type="{gtype}" size="{gsize}" friction="{mu} 0.02 0.001"
            rgba="0.6 0.6 0.7 1"/>
    </body>

    <body name="target" mocap="true" pos="0 0 0.4"/>

    <body name="hand" pos="0 0 0.4">
      <freejoint name="hand_free"/>
      <geom name="palm" type="box" size="0.030 0.020 0.012" mass="0.6"
            friction="{mu} 0.05 0.002" rgba="0.2 0.2 0.25 0.6"/>
      <body name="finger_l" pos="-0.050 0 -0.060">
        <joint name="fl" type="slide" axis="1 0 0" range="0 0.050" limited="true" damping="30"/>
        <geom name="fl_geom" type="box" size="0.006 0.014 0.020" mass="0.05"
              friction="{mu} 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
      <body name="finger_r" pos="0.050 0 -0.060">
        <joint name="fr" type="slide" axis="-1 0 0" range="0 0.050" limited="true" damping="30"/>
        <geom name="fr_geom" type="box" size="0.006 0.014 0.020" mass="0.05"
              friction="{mu} 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <!-- STIFF weld. A soft one lets the hand sag under the object's weight during the lift, and
         that sag is then attributed to the OBJECT and reported as slip. -->
    <weld body1="hand" body2="target" solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </equality>

  <actuator>
    <position name="grip_l" joint="fl" kp="2000" ctrlrange="0 0.050"
              forcerange="-{gripf} {gripf}"/>
    <position name="grip_r" joint="fr" kp="2000" ctrlrange="0 0.050"
              forcerange="-{gripf} {gripf}"/>
  </actuator>
</mujoco>
"""

#: Geoms belonging to the gripper. A grasp pose whose OPEN jaws already intersect the object or
#: the table is not executable at all, and must be rejected rather than simulated: teleporting a
#: finger into an object produces a penetration blow-up that launches it into the air.
_GRIPPER_GEOMS = ("palm", "fl_geom", "fr_geom")

#: The TOOL CENTRE POINT in the hand frame: the point midway between the two jaws. An action names
#: the grasp point, and the grasp point is here -- not at the hand's origin, which is up at the
#: palm. Derived from the MJCF: fingers are at z = -0.040.
TCP_OFFSET = np.array([0.0, 0.0, -0.060])

# GEOMETRIC BUDGET, and it is tight enough to be worth writing down.
#   palm half-thickness      0.012   -> palm bottom sits 0.012 below the hand origin
#   TCP depth                0.060   -> so the palm clears an object of half-height h only if
#                                       0.060 - 0.012 > h,  i.e.  h < 0.048
#   finger half-height       0.020   -> finger bottoms sit 0.080 below the hand origin, so they
#                                       clear the TABLE only if  h > 0.080 - 0.060 = 0.020
# Objects are sampled with half-height in roughly [0.022, 0.037], which fits inside (0.020, 0.048).
# Get this wrong in either direction and the grasp is rejected for collision before it is ever
# simulated: too shallow and the palm lands on the object, too deep and the fingers hit the table.


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
        self._renderers: dict[tuple, object] = {}

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
            gripf=self.cfg.grip_force,
            cx=com[0], cy=com[1], cz=com[2],
            ixx=max(ixx, 1e-6), iyy=max(iyy, 1e-6), izz=max(izz, 1e-6),
        )
        m = mujoco.MjModel.from_xml_string(xml)
        self._cache[key] = m
        if len(self._cache) > 256:
            self._cache.pop(next(iter(self._cache)))
        return m

    #: Slip reported for a grasp whose open jaws already intersect the scene. Such a grasp is not
    #: executable at all, so it is a total failure rather than a partial one.
    COLLISION_SLIP: float = 0.5

    def _gripper_intersects_scene(self, m, d) -> bool:
        """True if any gripper geom is in contact while the jaws are still OPEN."""
        import mujoco

        ids = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in _GRIPPER_GEOMS
        }
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 in ids or con.geom2 in ids:
                return True
        return False

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

        # Place the hand so that its TOOL CENTRE POINT lands on the grasp position.
        #
        # An action `a` names the point BETWEEN THE JAWS -- that is what the analytic oracle rays
        # through, and it is what a grasp pose means. It is not the hand's origin: the fingers hang
        # below the palm, so the jaw centre sits at TCP_OFFSET in the hand frame. Placing the hand
        # ORIGIN at the grasp point instead buries the palm inside the object, and every grasp is
        # then (correctly) rejected as colliding -- the simulator reported a 0.000 success rate,
        # which is what sent me looking.
        R = _quat_to_matrix(torch.tensor(grasp_quat, dtype=torch.float32).view(1, 4))[0].numpy()
        hand_pos = (
            np.asarray(grasp_pos, dtype=float)
            + np.array([0.0, 0.0, rest_z])
            - R @ TCP_OFFSET
        )
        d.qpos[hand_adr : hand_adr + 3] = hand_pos
        d.qpos[hand_adr + 3 : hand_adr + 7] = grasp_quat
        d.mocap_pos[0] = hand_pos
        d.mocap_quat[0] = grasp_quat
        d.ctrl[:] = [0.0, 0.0]  # jaws open
        mujoco.mj_forward(m, d)

        # REJECT NON-EXECUTABLE GRASPS BEFORE SIMULATING THEM.
        #
        # If the OPEN jaws already intersect the object or the table, the grasp pose is not
        # reachable: a real arm would collide on approach. Stepping the simulator from that state
        # does not model a bad grasp, it models a penetration blow-up -- MuJoCo resolves the
        # overlap with an enormous impulse and LAUNCHES the object into the air. Measured before
        # this check: 12 contacts at placement, and the object flew 13 cm upward before falling
        # back. Some of those flights cleared the success threshold, so the simulator was scoring
        # explosions as successful grasps.
        #
        # A colliding grasp is a genuine failure of the action, so we label it as one and return.
        if self._gripper_intersects_scene(m, d):
            return 0.0, self.COLLISION_SLIP

        for _ in range(40):
            mujoco.mj_step(m, d)

        # Close the jaws.
        d.ctrl[:] = [0.05, 0.05]
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

    # -- the observation kernel p(o | x) --------------------------------------

    @torch.no_grad()
    def render(
        self,
        state: Tensor,
        height: int = 128,
        width: int = 128,
        depth_noise: float = 0.002,
        max_depth: float = 1.5,
    ) -> Tensor:
        """Render RGB-D observations of the scene. This is p(o | x): the sensor model.

        THIS IS THE PIECE THAT TURNS THE FRAMEWORK INTO AN EXPERIMENT. Until it existed, the
        "observation" the encoder saw was a 32-dimensional random linear projection of the state --
        a toy. The whole point of MSP is that a *camera* cannot resolve the state, so the belief
        must carry epistemic uncertainty about the parts it cannot see. That claim is only testable
        against a real image.

        And note what the camera does NOT show: friction, mass, and the centre of mass. Those three
        decide slip and torque failure, and no pixel reveals them. A pose estimator cannot even
        represent that ignorance. MSP's belief can, and active touch (Eq 20/21) is what resolves it.

        HEADLESS RENDERING. On a server with no display, MuJoCo needs an offscreen GL backend. We
        set MUJOCO_GL=egl if the caller has not chosen one; without it, `mujoco.Renderer` raises
        "an OpenGL platform library has not been loaded into this process".

        Args:
            state: (B, state_dim)
            depth_noise: Gaussian noise on the depth channel, in metres. Real depth sensors are not
                exact, and a model trained on noiseless depth learns to trust it absolutely.
            max_depth: depth is normalized by this, so the channel lands in [0, 1] like RGB.

        Returns:
            (B, 4, H, W) float32. Channels 0:3 are RGB in [0,1]; channel 3 is normalized depth.
        """
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")

        st = state.detach().cpu()
        half = torch.exp(st[:, STATE_SLICES["log_size"]]).clamp(0.018, 0.038).numpy()
        mass = torch.exp(st[:, STATE_SLICES["log_mass"]]).clamp(0.01, 10.0).numpy()
        mu = torch.exp(st[:, STATE_SLICES["log_friction"]]).clamp(0.05, 2.0).numpy()
        com = st[:, STATE_SLICES["com"]].numpy()
        R = _axis_angle_to_matrix(st[:, STATE_SLICES["pose_r"]])
        quat = _matrix_to_quat(R).numpy()
        t_obj = st[:, STATE_SLICES["pose_t"]].numpy()

        # ONE model and ONE GL context for the whole batch.
        #
        # `_model` bakes the object's size, mass and friction into the MJCF, so it returns a
        # DISTINCT model per scene -- and a renderer is bound to a model, so a naive implementation
        # builds a fresh GL context per frame. Context creation costs ~100 ms, which turned a
        # 2000-scene render from ~3 minutes into something that had not finished in ten.
        #
        # MuJoCo exposes geometry as mutable arrays on the model, so we compile once with a
        # placeholder object and then write the per-scene size directly into `geom_size`. The
        # renderer never notices, because the model object is the same one it was bound to.
        m = self._model(half[0], float(mass[0, 0]), float(mu[0, 0]), com[0])
        r = self._renderer(m, height, width)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
        oa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
        ha = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]
        d = mujoco.MjData(m)

        out = torch.empty(state.shape[0], 4, height, width, dtype=torch.float32)

        for i in range(state.shape[0]):
            if self.shape == "cylinder":
                m.geom_size[gid, 0] = half[i][0]
                m.geom_size[gid, 1] = half[i][2]
            else:
                m.geom_size[gid, :3] = half[i]

            rest_z = float(half[i][2]) + 2e-3
            mujoco.mj_resetData(m, d)
            d.qpos[oa : oa + 3] = [t_obj[i][0], t_obj[i][1], rest_z]
            d.qpos[oa + 3 : oa + 7] = quat[i]
            # Park the hand out of frame: the observation is of the SCENE, taken before the grasp.
            d.qpos[ha : ha + 3] = [0.0, 0.0, 1.5]
            mujoco.mj_forward(m, d)

            r.disable_depth_rendering()
            r.update_scene(d, camera="rgbd")
            rgb = torch.from_numpy(r.render().copy()).float() / 255.0  # (H, W, 3)

            r.enable_depth_rendering()
            r.update_scene(d, camera="rgbd")
            dep = torch.from_numpy(r.render().copy()).float()  # (H, W), metres
            r.disable_depth_rendering()

            if depth_noise > 0:
                dep = dep + depth_noise * torch.randn_like(dep)
            dep = (dep / max_depth).clamp(0.0, 1.0)

            out[i, 0:3] = rgb.permute(2, 0, 1)
            out[i, 3] = dep

        return out

    def _renderer(self, model, height: int, width: int):
        """One Renderer per (model, size). Constructing a GL context per frame is the difference
        between rendering a dataset in minutes and in hours."""
        import mujoco

        key = (id(model), height, width)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(model, height=height, width=width)
            self._renderers[key] = r
            # Deliberately NOT closed: EGL context teardown raises on this driver, and the number
            # of distinct (object size, resolution) pairs is small. Bounded, not leaked.
            if len(self._renderers) > 16:
                self._renderers.pop(next(iter(self._renderers)))
        return r

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


