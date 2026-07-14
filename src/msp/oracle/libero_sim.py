"""M on real objects: LIBERO grocery meshes, MuJoCo rollouts, RGB-D from a ring of cameras.

THIS IS THE ORACLE THE PAPER SHOULD RUN ON, and the reason is not that the objects are prettier.

Boxes and cylinders cannot test the paper's central claim. Blueprint wrong-assumption #11 says:

    "Force closure computed on the estimated geometry is a valid success criterion." It is an
    analytic proxy evaluated on a HALLUCINATED SURFACE. A self-consistent but wrong reconstruction
    passes the check and fails the lift.

On a box, the "reconstruction" and the truth are the same object, so the gap between them is
identically zero and the claim is unfalsifiable. On a ketchup bottle it is not.

So the two tiers are deliberately given DIFFERENT geometry, and the difference is the experiment:

    ANALYTIC tier   plans on an oriented BOUNDING BOX fitted to the object -- roughly the fidelity
                    a pose-and-shape pipeline actually delivers, and exactly the kind of
                    self-consistent-but-wrong surface the paper indicts.
    SIMULATOR tier  rolls the grasp out against the object's TRUE convex decomposition (~20 boxes,
                    shipped with the asset), with its real mass and friction.
    CAMERA          sees the true textured mesh.

`CompositeOracle.tier_gap()` then measures how often force closure on the reconstruction passes and
the actual lift fails. That is a decisive experiment, and it needs no robot -- which matters,
because a simulation-only paper otherwise has no decisive experiment at all: with no hardware, M is
the ground truth by definition and "sufficiency with respect to M" is true by construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import Tensor

from msp.oracle.analytic import STATE_DIM, STATE_SLICES, _axis_angle_to_matrix, _quat_to_matrix
from msp.oracle.base import PhysicsOracle
from msp.oracle.libero_assets import LiberoObject, LiberoObjectLibrary
from msp.oracle.mujoco_sim import TCP_OFFSET, SimConfig, _matrix_to_quat
from msp.types import Outcome

__all__ = ["LiberoGraspOracle"]


_SCENE = """
<mujoco model="msp_libero">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          impratio="10" iterations="{iters}"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.35" rgb2="0.38 0.40 0.43"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    {object_assets}
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.85 0.85 0.85" castshadow="true"/>
    <light pos="-0.3 0.3 0.6" dir="0.3 -0.3 -0.6" diffuse="0.4 0.4 0.45" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" friction="1 0.02 0.001" material="grid_mat"/>

{cameras}

    <body name="object" pos="0 0 0.10">
      <freejoint name="obj"/>
      {object_geoms}
    </body>

    <body name="target" mocap="true" pos="0 0 0.4"/>
    <body name="hand" pos="0 0 0.4">
      <freejoint name="hand_free"/>
      <geom name="palm" type="box" size="0.030 0.020 0.012" mass="0.6"
            friction="1.0 0.05 0.002" rgba="0.2 0.2 0.25 0.6"/>
      <body name="finger_l" pos="-0.050 0 -0.060">
        <joint name="fl" type="slide" axis="1 0 0" range="0 0.050" limited="true" damping="30"/>
        <geom name="fl_geom" type="box" size="0.006 0.014 0.020" mass="0.05"
              friction="1.0 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
      <body name="finger_r" pos="0.050 0 -0.060">
        <joint name="fr" type="slide" axis="-1 0 0" range="0 0.050" limited="true" damping="30"/>
        <geom name="fr_geom" type="box" size="0.006 0.014 0.020" mass="0.05"
              friction="1.0 0.05 0.002" rgba="0.9 0.5 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <weld body1="hand" body2="target" solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <position name="grip_l" joint="fl" kp="2000" ctrlrange="0 0.050" forcerange="-{gripf} {gripf}"/>
    <position name="grip_r" joint="fr" kp="2000" ctrlrange="0 0.050" forcerange="-{gripf} {gripf}"/>
  </actuator>
</mujoco>
"""

_GRIPPER_GEOMS = ("palm", "fl_geom", "fr_geom")


@dataclass(frozen=True)
class SceneSpec:
    """One scene: which object, and the continuous state it is in."""

    object_index: int
    state: Tensor  # (STATE_DIM,)


class LiberoGraspOracle(PhysicsOracle):
    """Rigid-body grasp rollouts on real LIBERO objects, plus RGB-D from a ring of viewpoints.

    The continuous state `x` keeps the same 14 dimensions as the parametric world, so every
    diagnostic (J(x), ker J, the principal-angle test) works unchanged:

        pose_t[3]  pose_r[3] (yaw only; a table-top object rests flat)
        log_size[3]  -- a SCALE on the object's native size, nominal 0
        log_friction  log_mass  com[3]

    Object IDENTITY is a discrete scene variable, not part of x. That is correct: Theorem 4 is a
    LOCAL statement about the tangent space at a point of the state manifold, and a categorical
    object label has no tangent space. J(x) is computed per object.
    """

    differentiable = False  # you cannot backprop through mj_step
    N_VIEWS: int = 8

    def __init__(
        self,
        library: LiberoObjectLibrary | None = None,
        objects: list[str] | None = None,
        cfg: SimConfig | None = None,
        seed: int = 0,
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.lib = library or LiberoObjectLibrary()
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)

        # Default to the HOPE grocery set: cans, cartons, bottles, boxes.
        #
        # The wider LIBERO catalogue also contains BOWLS, WINE RACKS and BOOKS. They pass a
        # bounding-box width filter -- so a naive "is it narrower than the jaw" test admits them --
        # but they are not parallel-jaw targets at all: a rack has nothing to close on, and a bowl
        # is a thin shell the fingers slide off. Left in, they dominate the failure statistics and
        # make "real objects are hard to grasp" look like a physical finding rather than a sampling
        # mistake. (Measured: wine_rack, white_bowl and yellow_book each had a 100% analytic
        # false-positive rate, which is not a fact about geometry, it is a fact about the sampler.)
        pool = tuple(o for o in self.lib.graspable() if o.category == "stable_hope_objects")
        if not pool:
            pool = self.lib.graspable()
        if objects:
            want = set(objects)
            pool = tuple(o for o in self.lib.graspable() if o.name in want)
            if not pool:
                raise ValueError(f"none of {objects} are graspable LIBERO objects")
        # VALIDATE BY COMPILING, not by parsing.
        #
        # Two ways a LIBERO asset silently fails, and neither is visible from the XML alone:
        #
        #   * No resolvable visual mesh -> an INVISIBLE but perfectly solid object. The camera
        #     photographs an empty table while the physics grasps something. A corpus of pictures
        #     of nothing, with labels.
        #   * A hard-coded absolute texture path from the original author's machine -- several
        #     LIBERO assets contain
        #     `/home/yifengz/workspace/robosuite-master/.../ceramic.png` -- which raises only when
        #     MuJoCo actually compiles the model, i.e. deep inside a rollout, mid-corpus.
        #
        # So we build each object's model once, up front, and keep only the ones that compile.
        self._models: dict[int, mujoco.MjModel] = {}
        self._renderers: dict[tuple, mujoco.Renderer] = {}

        usable: list[LiberoObject] = []
        self.rejected: dict[str, str] = {}
        for o in pool:
            try:
                asset, geoms = LiberoObjectLibrary.body_xml(o)
                mujoco.MjModel.from_xml_string(
                    _SCENE.format(
                        timestep=self.cfg.timestep,
                        iters=self.cfg.solver_iterations,
                        gripf=self.cfg.grip_force,
                        object_assets=asset,
                        object_geoms=geoms,
                        cameras=self._cameras_xml(),
                    )
                )
                usable.append(o)
            except Exception as e:  # noqa: BLE001 - a broken asset must not kill the corpus
                self.rejected[o.name] = str(e).split("\n")[0][:120]

        if not usable:
            raise RuntimeError(
                "no LIBERO object compiled. Rejections:\n  "
                + "\n  ".join(f"{k}: {v}" for k, v in self.rejected.items())
            )
        self.objects: tuple[LiberoObject, ...] = tuple(usable)

    # -- properties ----------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def object_names(self) -> list[str]:
        return [o.name for o in self.objects]

    def base_half(self, object_index: Tensor) -> Tensor:
        """The bounding boxes the ANALYTIC tier plans on. (B, 3).

        This is the "reconstruction". It is crude on purpose -- see the module docstring.
        """
        return torch.tensor(
            np.stack([self.objects[int(i)].aabb_half_extents() for i in object_index]),
            dtype=torch.float32,
        )

    def provenance(self) -> dict[str, object]:
        return {
            "mujoco_version": mujoco.__version__,
            "asset_root": str(self.lib.root),
            "n_objects": self.n_objects,
            "objects": self.object_names(),
            **self.cfg.__dict__,
        }

    # -- model ---------------------------------------------------------------

    def _cameras_xml(self) -> str:
        import math

        out = []
        R, H = 0.24, 0.26
        for i in range(self.N_VIEWS):
            th = 2 * math.pi * i / self.N_VIEWS
            px, py, pz = R * math.cos(th), R * math.sin(th), H
            f = np.array([-px, -py, -pz], dtype=float)
            f /= np.linalg.norm(f)
            r = np.cross(f, np.array([0.0, 0.0, 1.0]))
            r /= np.linalg.norm(r)
            u = np.cross(r, f)
            out.append(
                f'    <camera name="view{i}" pos="{px:.4f} {py:.4f} {pz:.4f}" '
                f'xyaxes="{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {u[0]:.4f} {u[1]:.4f} {u[2]:.4f}" '
                f'fovy="48"/>'
            )
        return "\n".join(out)

    def _model(self, oi: int) -> mujoco.MjModel:
        if oi in self._models:
            return self._models[oi]
        asset, geoms = LiberoObjectLibrary.body_xml(self.objects[oi])
        xml = _SCENE.format(
            timestep=self.cfg.timestep,
            iters=self.cfg.solver_iterations,
            gripf=self.cfg.grip_force,
            object_assets=asset,
            object_geoms=geoms,
            cameras=self._cameras_xml(),
        )
        m = mujoco.MjModel.from_xml_string(xml)
        self._models[oi] = m
        return m

    def _renderer(self, m: mujoco.MjModel, h: int, w: int) -> mujoco.Renderer:
        key = (id(m), h, w)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(m, height=h, width=w)
            self._renderers[key] = r
            if len(self._renderers) > 24:  # GL contexts are bounded, not leaked
                self._renderers.pop(next(iter(self._renderers)))
        return r

    # -- scene sampling ------------------------------------------------------

    def sample_scenes(self, n: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """(object_index (n,), state (n, STATE_DIM)) with the pose the object ACTUALLY SETTLES IN.

        THE SETTLING IS NOT A DETAIL. A parametric box can be placed at a pose you choose: its
        resting height is closed-form and it stays where you put it. A scanned ketchup bottle
        cannot -- dropped on a table it topples, rolls, and comes to rest wherever gravity finds a
        stable configuration, very often NOT the yaw that was sampled.

        Sampling a nominal pose and then planning grasps against it means the analytic tier is
        planning for an object that is not there. Measured on the LIBERO objects, that alone had
        82% of grasp poses rejected for collision before they were ever simulated -- and it would
        have been written up as "real objects are hard to grasp" rather than as a bug.

        So we DROP the object, let the simulator find its resting pose, and read that pose back into
        the state. `pose_r` is an axis-angle, so it represents any orientation, including a bottle
        lying on its side.
        """
        import math

        g = generator
        oi = torch.randint(0, self.n_objects, (n,), generator=g)
        x = torch.zeros(n, STATE_DIM)
        xy = torch.randn(n, 2, generator=g) * 0.015
        yaw0 = (torch.rand(n, generator=g) * 2 - 1) * math.pi
        x[:, STATE_SLICES["log_size"]] = torch.randn(n, 3, generator=g) * 0.05
        x[:, STATE_SLICES["log_friction"]] = math.log(0.8) + torch.randn(n, 1, generator=g) * 0.3
        x[:, STATE_SLICES["log_mass"]] = math.log(0.25) + torch.randn(n, 1, generator=g) * 0.5
        x[:, STATE_SLICES["com"]] = torch.randn(n, 3, generator=g) * 0.004

        for i in range(n):
            pos, aa = self._settle(int(oi[i]), float(xy[i, 0]), float(xy[i, 1]), float(yaw0[i]))
            x[i, STATE_SLICES["pose_t"]] = torch.from_numpy(pos).float()
            x[i, STATE_SLICES["pose_r"]] = torch.from_numpy(aa).float()
        return oi, x

    def _settle(self, oi: int, px: float, py: float, yaw: float):
        """Drop the object; return the pose it comes to rest in: (pos[3], axis_angle[3])."""
        m = self._model(oi)
        d = mujoco.MjData(m)
        oa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
        ha = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]
        d.qpos[oa : oa + 3] = [px, py, 0.12]
        d.qpos[oa + 3 : oa + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qpos[ha : ha + 3] = [0.0, 0.0, 1.5]
        mujoco.mj_forward(m, d)
        for _ in range(700):
            mujoco.mj_step(m, d)
        pos = d.qpos[oa : oa + 3].copy()
        q = d.qpos[oa + 3 : oa + 7].copy()
        q = q / (np.linalg.norm(q) + 1e-9)
        angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
        s = np.sqrt(max(1.0 - q[0] ** 2, 1e-12))
        axis = q[1:4] / s if s > 1e-6 else np.array([0.0, 0.0, 1.0])
        return pos, (axis * angle)

    # -- the operator --------------------------------------------------------

    def _rollout(self, oi: int, x: np.ndarray, action: np.ndarray) -> tuple[float, float]:
        m = self._model(oi)
        d = mujoco.MjData(m)
        c = self.cfg

        oa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
        ha = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]

        # The state already carries the SETTLED pose (see `sample_scenes`), so place the object
        # there directly. A second drop would land it somewhere else, and the action -- which was
        # planned against the settled pose -- would miss.
        R_obj = _axis_angle_to_matrix(torch.tensor(x[3:6], dtype=torch.float32).view(1, 3))
        q = _matrix_to_quat(R_obj).numpy()[0]
        d.qpos[oa : oa + 3] = np.asarray(x[0:3], dtype=float)
        d.qpos[oa + 3 : oa + 7] = q
        d.qpos[ha : ha + 3] = [0.0, 0.0, 1.5]
        d.ctrl[:] = [0.0, 0.0]
        mujoco.mj_forward(m, d)
        for _ in range(60):
            mujoco.mj_step(m, d)
        rest = d.qpos[oa : oa + 3].copy()

        # Place the gripper so its TOOL CENTRE POINT lands on the grasp point.
        #
        # THE ACTION IS A GRASP POINT IN THE WORLD FRAME, NOT AN OFFSET FROM THE OBJECT.
        #
        # `sample_actions` builds it as `t_obj + lift`, where t_obj = x[0:3] is the object's SETTLED
        # WORLD position (`sample_scenes` writes the settled pose straight into the state). Adding
        # `rest` -- which is that same world position, re-read after a brief re-settle -- counts the
        # object's position TWICE:
        #
        #     hand_pos = rest + (t_obj + lift) - R@TCP  =  2*t_obj + lift - R@TCP
        #
        # The gripper was therefore displaced by the object's entire world position: several cm up
        # in z, and sideways in x/y. What that produced looked exactly like physics, which is why it
        # survived:
        #
        #   * The taller the object, the larger the z error, so the jaws closed on empty air and the
        #     object was left standing on the table. Slip is measured against the hand, so an
        #     ungrasped object reports slip == lift_height -- and SIX objects reported a mean slip of
        #     0.1185 m against a lift_height of 0.12. Identical "physics" across six different
        #     geometries is not physics.
        #   * Only ketchup and salad_dressing worked (42%, 26%) -- the two SHORTEST objects, i.e. the
        #     smallest error.
        #   * The NOMINAL grasp (zero proposal noise) was the WORST of all, 5%: it is deterministically
        #     displaced, whereas sampling noise sometimes cancels the offset. A proposal whose mode is
        #     its worst sample is not a proposal, it is a bug.
        #   * And because the analytic tier plans correctly in the world frame, it was scoring one
        #     grasp while the simulator executed another -- which is precisely how one measures a
        #     grasp-quality metric to be "uninformative" (AUC 0.539) when it is nothing of the kind.
        #
        # The object may drift a hair during the 60-step re-settle above, so the grasp follows it by
        # (rest - x[0:3]); that term is ~0 for an already-settled object and is NOT a second offset.
        gq = action[3:7] / (np.linalg.norm(action[3:7]) + 1e-9)
        R = _quat_to_matrix(torch.tensor(gq, dtype=torch.float32).view(1, 4))[0].numpy()
        drift = rest - np.asarray(x[0:3], dtype=float)
        hand_pos = np.asarray(action[0:3], dtype=float) + drift - R @ TCP_OFFSET
        d.qpos[ha : ha + 3] = hand_pos
        d.qpos[ha + 3 : ha + 7] = gq
        d.mocap_pos[0] = hand_pos
        d.mocap_quat[0] = gq
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        # A grasp pose whose OPEN jaws already intersect the scene is not executable.
        ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in _GRIPPER_GEOMS}
        for i in range(d.ncon):
            if d.contact[i].geom1 in ids or d.contact[i].geom2 in ids:
                return 0.0, 0.5

        for _ in range(40):
            mujoco.mj_step(m, d)

        d.ctrl[:] = [0.05, 0.05]
        for _ in range(c.close_steps):
            mujoco.mj_step(m, d)
        for _ in range(c.settle_steps):
            mujoco.mj_step(m, d)

        grasped = d.qpos[oa : oa + 3].copy()
        hand0 = d.qpos[ha : ha + 3].copy()
        mocap0 = d.mocap_pos[0].copy()

        dz = c.lift_speed * c.timestep
        for i in range(c.lift_steps):
            d.mocap_pos[0][2] = min(mocap0[2] + dz * i, mocap0[2] + c.lift_height)
            mujoco.mj_step(m, d)

        final = d.qpos[oa : oa + 3].copy()
        success = float(final[2] > rest[2] + c.success_height)
        travel = d.qpos[ha : ha + 3] - hand0
        slip = float(np.linalg.norm(final - (grasped + travel)))
        return success, min(slip, 0.5)

    @torch.no_grad()
    def query_scenes(
        self, object_index: Tensor, state: Tensor, actions: Tensor
    ) -> Outcome:
        """y ~ M(. | x, a) on real meshes. (B,), (B, d_X), (B, Na, 7) -> Outcome (B, Na, 1)."""
        b, na, _ = actions.shape
        oi = object_index.cpu().numpy()
        xs = state.detach().cpu().numpy()
        acts = actions.detach().cpu().numpy()

        succ = np.zeros((b, na), dtype=np.float32)
        slip = np.zeros((b, na), dtype=np.float32)
        for i in range(b):
            for j in range(na):
                succ[i, j], slip[i, j] = self._rollout(int(oi[i]), xs[i], acts[i, j])

        out = Outcome(
            succ=torch.from_numpy(succ).unsqueeze(-1).to(state.device),
            margin=torch.zeros(b, na, 1, device=state.device),  # the analytic tier's job
            slip=torch.from_numpy(slip).unsqueeze(-1).to(state.device),
        )
        out.validate()
        return out

    def query(self, state: Tensor, actions: Tensor) -> Outcome:  # pragma: no cover
        raise NotImplementedError(
            "LiberoGraspOracle needs the object identity: call query_scenes(object_index, state, "
            "actions). Object identity is a discrete scene variable, not part of the continuous "
            "state x, because Theorem 4 is a statement about a tangent space and a categorical "
            "label has none."
        )

    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "Not differentiable -- you cannot autodiff through mj_step. Phi (and therefore J(x), "
            "and therefore contribution C2) comes from the ANALYTIC tier. See CompositeOracle."
        )

    # -- the observation kernel ----------------------------------------------

    @torch.no_grad()
    def render(
        self,
        object_index: Tensor,
        state: Tensor,
        height: int = 96,
        width: int = 96,
        view: int = 0,
        depth_noise: float = 0.002,
        max_depth: float = 1.5,
    ) -> Tensor:
        """RGB-D of the settled scene. (B, 4, H, W).

        The camera sees the TRUE textured mesh -- not the convex decomposition the physics uses,
        and not the bounding box the analytic planner uses. Three different geometries, on purpose.
        """
        if not 0 <= view < self.N_VIEWS:
            raise ValueError(f"view must be in [0, {self.N_VIEWS}); got {view}")
        cam = f"view{view}"

        oi = object_index.cpu().numpy()
        xs = state.detach().cpu().numpy()
        out = torch.empty(state.shape[0], 4, height, width, dtype=torch.float32)

        for i in range(state.shape[0]):
            m = self._model(int(oi[i]))
            d = mujoco.MjData(m)
            oa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
            ha = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]

            R_obj = _axis_angle_to_matrix(torch.tensor(xs[i][3:6], dtype=torch.float32).view(1, 3))
            q = _matrix_to_quat(R_obj).numpy()[0]
            d.qpos[oa : oa + 3] = xs[i][0:3]
            d.qpos[oa + 3 : oa + 7] = q
            d.qpos[ha : ha + 3] = [0.0, 0.0, 1.5]  # hand out of frame: this is the SCENE
            mujoco.mj_forward(m, d)
            for _ in range(60):
                mujoco.mj_step(m, d)

            r = self._renderer(m, height, width)
            r.disable_depth_rendering()
            r.update_scene(d, camera=cam)
            rgb = torch.from_numpy(r.render().copy()).float() / 255.0

            r.enable_depth_rendering()
            r.update_scene(d, camera=cam)
            dep = torch.from_numpy(r.render().copy()).float()
            r.disable_depth_rendering()

            if depth_noise > 0:
                dep = dep + depth_noise * torch.randn_like(dep)
            out[i, 0:3] = rgb.permute(2, 0, 1)
            out[i, 3] = (dep / max_depth).clamp(0.0, 1.0)

        return out
