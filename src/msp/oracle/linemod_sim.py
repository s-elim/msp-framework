"""Grasp outcomes for LINEMOD objects: the BOP LM meshes, rolled out in MuJoCo.

The LINEMOD benchmark supplies REAL RGB-D images with ground-truth 6D poses of 15 scanned
objects. It supplies no grasp outcomes. This module produces them: every candidate grasp is
transformed into the object frame with the ground-truth pose, executed by a floating
parallel-jaw gripper against a convex decomposition of the object's scanned mesh, and scored by
whether the object is lifted.

So the observation is real and the outcome is simulated. The paper states this split wherever a
LINEMOD number appears.

Conventions
    * LM models are in millimetres with the origin at the bounding-box centre and +z pointing up
      from the board the objects stand on (verified on the test poses: see
      `scripts/sr/check_lm_upright.py`). The simulator places the object upright on a table at
      z = 0 and lets it settle.
    * A grasp is a 4x4 TCP pose. The hand's local x axis is the closing direction and the
      approach direction is the hand's local -z, as in `libero_sim`.
    * Physical parameters (friction, mass, centre of mass) are sampled per scene and are not
      observable in the image, by construction.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["GripperSpec", "LinemodGraspSim", "LinemodObject", "PhysParams", "SimSpec"]

#: The 13 objects of the standard LINEMOD protocol. Bowl (3) and cup (7) are excluded by that
#: protocol because their original meshes were incomplete.
LM13 = (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)
LM_NAMES = {
    1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can", 6: "cat", 7: "cup",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron",
    14: "lamp", 15: "phone",
}
#: BOP lists discrete symmetries for these: ADD-S is the pose metric for them.
LM_SYMMETRIC = (3, 10, 11)


@dataclass(frozen=True)
class GripperSpec:
    """A two-finger parallel gripper with a 140 mm class stroke.

    The stroke is chosen so that every LM13 object has some graspable width; a 85 mm class
    stroke leaves the benchvise, iron and lamp with almost no executable grasp.
    """

    half_opening: float = 0.075  # finger centre to palm centre when fully open, metres
    finger_half: tuple[float, float, float] = (0.006, 0.012, 0.035)
    palm_half: tuple[float, float, float] = (0.085, 0.020, 0.012)
    finger_offset_z: float = -0.085  # finger centre below the palm
    grip_force: float = 10.0  # N per finger

    @property
    def tcp_offset(self) -> np.ndarray:
        """TCP (between the finger pads) in the hand frame."""
        return np.array([0.0, 0.0, self.finger_offset_z])


@dataclass(frozen=True)
class SimSpec:
    timestep: float = 0.002
    solver_iterations: int = 60
    settle_steps: int = 400  # object settling on the table
    close_steps: int = 300
    hold_steps: int = 100
    lift_steps: int = 400
    lift_height: float = 0.12
    lift_speed: float = 0.15
    success_height: float = 0.05


@dataclass(frozen=True)
class PhysParams:
    friction: float
    mass: float
    com: tuple[float, float, float]  # metres, object frame

    def to_array(self) -> np.ndarray:
        return np.array([self.friction, self.mass, *self.com], dtype=np.float32)


@dataclass(frozen=True)
class LinemodObject:
    obj_id: int
    name: str
    parts: tuple[str, ...]  # absolute paths of convex parts (OBJ, metres)
    half_extents: tuple[float, float, float]  # metres, model frame
    diameter: float  # metres
    volume: float  # m^3 of the watertight hull union (approximate)

    @property
    def nominal_mass(self) -> float:
        """Volume times 500 kg/m^3, clipped. LM masses are not published; this is a prior."""
        return float(np.clip(self.volume * 500.0, 0.05, 2.0))


def decompose_lm_models(lm_root: Path, cache: Path, obj_ids: tuple[int, ...] = LM13,
                        threshold: float = 0.05, max_parts: int = 24) -> list[LinemodObject]:
    """Convex-decompose each LM mesh once with CoACD and cache the parts as OBJ files."""
    import trimesh

    cache.mkdir(parents=True, exist_ok=True)
    info = json.loads((lm_root / "models" / "models_info.json").read_text())
    out: list[LinemodObject] = []
    for oid in obj_ids:
        d = cache / f"obj_{oid:06d}"
        meta_p = d / "meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
        else:
            import coacd

            d.mkdir(parents=True, exist_ok=True)
            mesh = trimesh.load(lm_root / "models" / f"obj_{oid:06d}.ply", force="mesh")
            mesh.apply_scale(1e-3)  # mm -> m
            cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
            parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_parts,
                                    seed=0)
            files = []
            vol = 0.0
            for i, (v, f) in enumerate(parts):
                pm = trimesh.Trimesh(v, f)
                p = d / f"part_{i:02d}.obj"
                pm.export(p)
                files.append(str(p.resolve()))
                vol += float(abs(pm.volume))
            meta = {"parts": files, "volume_parts": vol,
                    "volume_mesh": float(abs(mesh.volume)) if mesh.is_watertight else None}
            meta_p.write_text(json.dumps(meta, indent=1))
            log.info("obj %d: %d convex parts", oid, len(files))
        mi = info[str(oid)]
        vol = meta["volume_mesh"] or meta["volume_parts"]
        out.append(LinemodObject(
            obj_id=oid, name=LM_NAMES[oid], parts=tuple(meta["parts"]),
            half_extents=(mi["size_x"] / 2e3, mi["size_y"] / 2e3, mi["size_z"] / 2e3),
            diameter=mi["diameter"] / 1e3, volume=vol,
        ))
    return out


_SCENE = """
<mujoco model="lm_grasp">
  <compiler angle="radian"/>
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic" impratio="10"
          iterations="{iters}"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <asset>
{meshes}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" friction="1 0.02 0.001"/>
    <body name="object" pos="0 0 0.2">
      <freejoint name="obj"/>
      <inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>
{geoms}
    </body>
    <body name="target" mocap="true" pos="0 0 1.0"/>
    <body name="hand" pos="0 0 1.0">
      <freejoint name="hand_free"/>
      <geom name="palm" type="box" size="{ph[0]} {ph[1]} {ph[2]}" mass="0.6"
            friction="1.0 0.05 0.002"/>
      <body name="finger_l" pos="-{ho} 0 {fz}">
        <joint name="fl" type="slide" axis="1 0 0" range="0 {ho}" limited="true" damping="30"/>
        <geom name="fl_geom" type="box" size="{fh[0]} {fh[1]} {fh[2]}" mass="0.05"
              friction="1.0 0.05 0.002"/>
      </body>
      <body name="finger_r" pos="{ho} 0 {fz}">
        <joint name="fr" type="slide" axis="-1 0 0" range="0 {ho}" limited="true" damping="30"/>
        <geom name="fr_geom" type="box" size="{fh[0]} {fh[1]} {fh[2]}" mass="0.05"
              friction="1.0 0.05 0.002"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="hand" body2="target" solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <position name="grip_l" joint="fl" kp="2000" ctrlrange="0 {ho}" forcerange="-{gf} {gf}"/>
    <position name="grip_r" joint="fr" kp="2000" ctrlrange="0 {ho}" forcerange="-{gf} {gf}"/>
  </actuator>
</mujoco>
"""

_HAND_GEOMS = ("palm", "fl_geom", "fr_geom")


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


class LinemodGraspSim:
    """One MuJoCo model per object; rollouts are independent and deterministic given inputs."""

    def __init__(self, obj: LinemodObject, gripper: GripperSpec | None = None,
                 sim: SimSpec | None = None) -> None:
        self.obj = obj
        self.g = gripper or GripperSpec()
        self.s = sim or SimSpec()
        meshes = "\n".join(
            f'    <mesh name="p{i}" file="{p}"/>' for i, p in enumerate(obj.parts))
        geoms = "\n".join(
            f'      <geom name="og{i}" type="mesh" mesh="p{i}" mass="0" density="0" '
            f'friction="0.6 0.005 0.0001"/>' for i in range(len(obj.parts)))
        xml = _SCENE.format(dt=self.s.timestep, iters=self.s.solver_iterations, meshes=meshes,
                            geoms=geoms, ph=self.g.palm_half, fh=self.g.finger_half,
                            ho=self.g.half_opening, fz=self.g.finger_offset_z,
                            gf=self.g.grip_force)
        self.model = mujoco.MjModel.from_xml_string(xml)
        m = self.model
        self._oa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
        self._ha = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]
        self._obody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._ogeoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"og{i}")
                        for i in range(len(obj.parts))]
        self._hand_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                          for n in _HAND_GEOMS}
        self._floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hx, hy, hz = obj.half_extents
        # Inertia of a solid box with the object's extents, scaled by mass at rollout time.
        self._unit_inertia = np.array([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2]) / 3.0
        self._rest_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    # -- physics parameters ---------------------------------------------------------------

    def sample_phys(self, rng: np.random.Generator) -> PhysParams:
        hx, hy, hz = self.obj.half_extents
        return PhysParams(
            friction=float(np.exp(np.log(0.5) + 0.4 * rng.standard_normal())),
            mass=float(np.exp(np.log(self.obj.nominal_mass) + 0.4 * rng.standard_normal())),
            com=tuple(float(v) for v in 0.15 * np.array([hx, hy, hz]) * rng.standard_normal(3)),
        )

    def nominal_phys(self) -> PhysParams:
        return PhysParams(friction=0.5, mass=self.obj.nominal_mass, com=(0.0, 0.0, 0.0))

    def _apply_phys(self, p: PhysParams) -> None:
        m = self.model
        m.body_mass[self._obody] = p.mass
        m.body_inertia[self._obody] = p.mass * self._unit_inertia
        m.body_ipos[self._obody] = np.asarray(p.com)
        for gid in self._ogeoms:
            m.geom_friction[gid, 0] = p.friction

    # -- resting pose ----------------------------------------------------------------------

    def rest_pose(self, p: PhysParams) -> tuple[np.ndarray, np.ndarray]:
        """World pose (pos, quat) the upright object settles into on the table."""
        key = (round(p.mass, 4), tuple(round(c, 5) for c in p.com), round(p.friction, 3))
        if key in self._rest_cache:
            return self._rest_cache[key]
        self._apply_phys(p)
        d = mujoco.MjData(self.model)
        oa, ha = self._oa, self._ha
        d.qpos[oa:oa + 3] = [0.0, 0.0, self.obj.half_extents[2] + 0.002]
        d.qpos[oa + 3:oa + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[ha:ha + 3] = [0.0, 0.0, 2.0]
        d.mocap_pos[0] = [0.0, 0.0, 2.0]
        mujoco.mj_forward(self.model, d)
        for _ in range(self.s.settle_steps):
            mujoco.mj_step(self.model, d)
        pos = d.qpos[oa:oa + 3].copy()
        q = d.qpos[oa + 3:oa + 7].copy()
        out = (pos, q / np.linalg.norm(q))
        if len(self._rest_cache) < 4096:
            self._rest_cache[key] = out
        return out

    # -- one grasp -------------------------------------------------------------------------

    def rollout(self, T_obj_tcp: np.ndarray, p: PhysParams) -> tuple[float, float, bool]:
        """Execute one grasp given as a TCP pose in the OBJECT frame.

        Returns (success, slip [m], executable). A grasp whose open jaws already intersect the
        object or the table is not executable and counts as a failure.
        """
        rest_pos, rest_q = self.rest_pose(p)
        self._apply_phys(p)
        m = self.model
        d = mujoco.MjData(m)
        oa, ha = self._oa, self._ha
        d.qpos[oa:oa + 3] = rest_pos
        d.qpos[oa + 3:oa + 7] = rest_q
        R_wo = _matrix_from_quat(rest_q)
        R_wt = R_wo @ T_obj_tcp[:3, :3]
        p_wt = rest_pos + R_wo @ T_obj_tcp[:3, 3]
        hand_pos = p_wt - R_wt @ self.g.tcp_offset
        hq = _quat_from_matrix(R_wt)
        d.qpos[ha:ha + 3] = hand_pos
        d.qpos[ha + 3:ha + 7] = hq
        d.mocap_pos[0] = hand_pos
        d.mocap_quat[0] = hq
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self._hand_ids or c.geom2 in self._hand_ids:
                return 0.0, 0.0, False

        for _ in range(20):
            mujoco.mj_step(m, d)
        d.ctrl[:] = [self.g.half_opening, self.g.half_opening]
        for _ in range(self.s.close_steps + self.s.hold_steps):
            mujoco.mj_step(m, d)
        grasped = d.qpos[oa:oa + 3].copy()
        hand0 = d.qpos[ha:ha + 3].copy()
        z0 = d.mocap_pos[0][2]
        dz = self.s.lift_speed * self.s.timestep
        for i in range(self.s.lift_steps):
            d.mocap_pos[0][2] = min(z0 + dz * i, z0 + self.s.lift_height)
            mujoco.mj_step(m, d)
        final = d.qpos[oa:oa + 3].copy()
        success = float(final[2] > rest_pos[2] + self.s.success_height)
        travel = d.qpos[ha:ha + 3] - hand0
        slip = float(min(np.linalg.norm(final - (grasped + travel)), 0.5))
        return success, slip, True

    def provenance(self) -> dict:
        return {"mujoco": mujoco.__version__, "gripper": asdict(self.g), "sim": asdict(self.s),
                "object": self.obj.name, "n_parts": len(self.obj.parts),
                "nominal_mass": self.obj.nominal_mass,
                "MUJOCO_GL": os.environ.get("MUJOCO_GL", "")}
