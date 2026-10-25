"""MuJoCo renders of LM-Grasp rollouts for the method figure.

The renderer is a lit twin of the simulator model (same XML plus a shadow-casting light), so a
qpos recorded by `LinemodGraspSim.rollout(..., trace=...)` renders unchanged. The floor is white
so only shadows show, and background pixels (no geometry) are painted white.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from common import REPO

LIGHT = ('<light directional="true" pos="0.4 -0.6 2.0" dir="-0.2 0.3 -1" castshadow="true" '
         'diffuse="0.60 0.60 0.60" ambient="0.30 0.30 0.30" specular="0.1 0.1 0.1"/>')
VISUAL = ('<visual><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" '
          'diffuse="0.35 0.35 0.35" specular="0 0 0"/></visual>')
OBJ_RGBA = (0.88, 0.86, 0.82, 1.0)
FLOOR_RGBA = (1.0, 1.0, 1.0, 1.0)  # lit floor saturates to white; shadows stay


def sim_for(obj: int):  # noqa: ANN201
    from msp.oracle.linemod_sim import LinemodGraspSim, decompose_lm_models

    return LinemodGraspSim(decompose_lm_models(REPO / "data/lm", REPO / "data/lm_grasp/coacd",
                                               obj_ids=(obj,))[0])


def side_view(sim, p, T_obj_tcp: np.ndarray) -> tuple[np.ndarray, float]:  # noqa: ANN001
    """Look-at point (world) and azimuth (deg) of a camera that sees the grasp's closing axis
    side on: the view direction is the hand's y axis on the settled object."""
    pos, q = sim.rest_pose(p)
    R_wo = np.empty(9)
    mujoco.mju_quat2Mat(R_wo, q)
    y_w = R_wo.reshape(3, 3) @ T_obj_tcp[:3, 1]
    return pos + np.array([0.0, 0.0, 0.10]), float(np.degrees(np.arctan2(y_w[1], y_w[0])))


def _rgba(hex_colour: str) -> tuple[float, float, float, float]:
    h = hex_colour.lstrip("#")
    return (*(int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)), 1.0)


class SimRenderer:
    def __init__(self, sim, size: int = 320) -> None:  # noqa: ANN001
        xml = sim.xml.replace("<worldbody>", "<worldbody>\n    " + LIGHT, 1)
        xml = xml.replace("<asset>", VISUAL + "\n  <asset>", 1)
        # an infinite floor leaves no horizon edge in the frame
        xml = xml.replace('size="2 2 0.05"', 'size="0 0 0.05"', 1)
        self.m = mujoco.MjModel.from_xml_string(xml)
        assert self.m.nq == sim.model.nq, "render twin changed the qpos layout"
        self.d = mujoco.MjData(self.m)
        name = {i: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i)
                for i in range(self.m.ngeom)}
        self.hand = [i for i, n in name.items() if n in ("palm", "fl_geom", "fr_geom")]
        for i, n in name.items():
            if n == "floor":
                self.m.geom_rgba[i] = FLOOR_RGBA
            elif n and n.startswith("og"):
                self.m.geom_rgba[i] = OBJ_RGBA
        self.r = mujoco.Renderer(self.m, size, size)

    def render(self, qpos: np.ndarray, hand: str, lookat: np.ndarray, distance: float,
               azimuth: float, elevation: float) -> np.ndarray:
        """(size, size, 3) uint8 image of state `qpos` with the hand in colour `hand`."""
        for i in self.hand:
            self.m.geom_rgba[i] = _rgba(hand)
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        self.r.update_scene(self.d, cam)
        rgb = self.r.render().copy()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        rgb[seg[..., 0] < 0] = 255
        return rgb
