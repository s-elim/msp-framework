"""LM-Grasp: real LINEMOD RGB-D observations paired with simulated grasp outcomes.

Each scene is one REAL test image of one LM13 object (BOP `lm` test split). The pipeline:

    1. crop       a square RGB-D crop around the visible-mask bounding box (the usual "known
                  detection" assumption of LINEMOD pose papers), resized to S x S.
    2. propose    N_a candidate grasps built ONLY from the observation: a visible surface point,
                  an approach direction tilted from straight down onto the table plane fitted to
                  the depth image, a penetration depth and a closing-axis angle. No ground truth
                  enters the proposal.
    3. execute    each candidate is mapped into the object frame with the ground-truth pose and
                  rolled out by `LinemodGraspSim` with per-scene friction, mass and centre of
                  mass drawn from a prior. Those three are invisible in the image.

Actions are expressed in the camera frame relative to the visible-point centroid c, as
[t - c (3), first two columns of R (6)]; the 6-D rotation is continuous (Zhou et al. 2019).

Splits are made per object over BLOCKS of 50 consecutive image ids, so near-duplicate
neighbouring frames do not straddle train and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["LMIndexEntry", "Proposal", "crop_observation", "index_lm", "propose_grasps",
           "split_of", "tcp_pose"]

SPLITS = ("train", "val", "calib", "test")
SPLIT_P = (0.55, 0.10, 0.15, 0.20)
BLOCK = 50


@dataclass(frozen=True)
class LMIndexEntry:
    obj_id: int
    im_id: int
    R: np.ndarray  # (3,3) model -> camera
    t: np.ndarray  # (3,) metres
    K: np.ndarray  # (3,3)
    bbox_visib: tuple[int, int, int, int]
    visib_fract: float


def index_lm(root: Path, obj_ids: tuple[int, ...]) -> list[LMIndexEntry]:
    out = []
    for oid in obj_ids:
        sd = root / "test" / f"{oid:06d}"
        gt = json.loads((sd / "scene_gt.json").read_text())
        cam = json.loads((sd / "scene_camera.json").read_text())
        info = json.loads((sd / "scene_gt_info.json").read_text())
        for k in sorted(gt, key=int):
            g = gt[k][0]
            if g["obj_id"] != oid:
                continue
            out.append(LMIndexEntry(
                obj_id=oid, im_id=int(k),
                R=np.array(g["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                t=np.array(g["cam_t_m2c"], dtype=np.float64) / 1000.0,
                K=np.array(cam[k]["cam_K"], dtype=np.float64).reshape(3, 3),
                bbox_visib=tuple(int(v) for v in info[k][0]["bbox_visib"]),
                visib_fract=float(info[k][0]["visib_fract"]),
            ))
    return out


def split_of(obj_id: int, im_id: int, seed: int = 0) -> str:
    """Deterministic block split: all frames of a 50-frame block share a split."""
    rng = np.random.default_rng([seed, obj_id, im_id // BLOCK])
    return SPLITS[int(rng.choice(4, p=SPLIT_P))]


def _load(root: Path, e: LMIndexEntry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = root / "test" / f"{e.obj_id:06d}"
    rgb = np.asarray(Image.open(sd / "rgb" / f"{e.im_id:06d}.png").convert("RGB"))
    dep = np.asarray(Image.open(sd / "depth" / f"{e.im_id:06d}.png"), dtype=np.float32) / 1000.0
    msk = np.asarray(Image.open(sd / "mask_visib" / f"{e.im_id:06d}_000000.png")) > 0
    return rgb, dep, msk


def crop_observation(root: Path, e: LMIndexEntry, size: int = 96, pad: float = 1.4
                     ) -> dict[str, np.ndarray]:
    """RGB-D crop plus the observation-derived geometry the proposal and the actions use.

    Returns obs (4, S, S) float16 [RGB in [0,1], depth relative to the object's median depth,
    divided by 0.25 m and clipped to [-1, 1]], visible points (M, 3) in metres, centroid c,
    and crop meta (u0, v0, side) for projecting actions back into the crop.
    """
    rgb, dep, msk = _load(root, e)
    x, y, w, h = e.bbox_visib
    side = int(max(32, pad * max(w, h)))
    cu, cv = x + w / 2.0, y + h / 2.0
    u0, v0 = int(round(cu - side / 2)), int(round(cv - side / 2))
    H, W = dep.shape
    # pad the image so crops near the border stay square
    P = side
    rgb_p = np.pad(rgb, ((P, P), (P, P), (0, 0)))
    dep_p = np.pad(dep, ((P, P), (P, P)))
    rc = rgb_p[v0 + P:v0 + P + side, u0 + P:u0 + P + side]
    dc = dep_p[v0 + P:v0 + P + side, u0 + P:u0 + P + side]
    rc = np.asarray(Image.fromarray(rc).resize((size, size), Image.BILINEAR), np.float32) / 255
    dc = np.asarray(Image.fromarray(dc).resize((size, size), Image.NEAREST), np.float32)

    vs, us = np.nonzero(msk & (dep > 0))
    z = dep[vs, us]
    K = e.K
    pts = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
    d_med = float(np.median(z)) if len(z) else float(e.t[2])
    drel = np.where(dc > 0, np.clip((dc - d_med) / 0.25, -1.0, 1.0), 0.0)
    obs = np.concatenate([rc.transpose(2, 0, 1), drel[None]], 0).astype(np.float16)
    c = pts.mean(0) if len(pts) else e.t.copy()
    n_tab, h_tab = table_plane(dep, msk, e.bbox_visib, K, c,
                               np.random.default_rng([e.obj_id, e.im_id, 3]))
    return {"obs": obs, "points": pts, "c": c, "crop": np.array([u0, v0, side], np.float32),
            "d_med": np.float32(d_med), "n_table": n_tab.astype(np.float32),
            "h_table": np.float32(h_tab)}


def table_plane(dep: np.ndarray, msk: np.ndarray, bbox: tuple[int, int, int, int],
                K: np.ndarray, c: np.ndarray, rng: np.random.Generator,
                iters: int = 150, tol: float = 0.004) -> tuple[np.ndarray, float]:
    """RANSAC plane through the depth pixels in a ring around the object (the supporting board).

    Returns the unit normal pointing toward the camera side of the plane and the signed height
    of the visible centroid c above the plane. Uses only the depth image and the detection mask.
    """
    x, y, w, h = bbox
    pad = int(0.6 * max(w, h))
    H, W = dep.shape
    ring = np.zeros_like(msk)
    ring[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)] = True
    grow = np.zeros_like(msk)
    grow[max(0, y - 4):min(H, y + h + 4), max(0, x - 4):min(W, x + w + 4)] = True
    ring &= ~grow & (dep > 0)
    vs, us = np.nonzero(ring)
    fallback = (np.array([0.0, -1.0, 0.0]), 0.0)
    if len(vs) < 100:
        return fallback
    sel = rng.choice(len(vs), min(3000, len(vs)), replace=False)
    z = dep[vs[sel], us[sel]]
    P = np.stack([(us[sel] - K[0, 2]) * z / K[0, 0], (vs[sel] - K[1, 2]) * z / K[1, 1], z], 1)
    best, best_n, best_p = 0, None, None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = int((np.abs((P - s[0]) @ n) < tol).sum())
        if cnt > best:
            best, best_n, best_p = cnt, n, s[0]
    if best_n is None or best < 0.3 * len(P):
        return fallback
    if best_n @ (-best_p) < 0:  # orient toward the camera
        best_n = -best_n
    return best_n, float((c - best_p) @ best_n)


@dataclass(frozen=True)
class Proposal:
    T: np.ndarray  # (Na, 4, 4) TCP poses in the camera frame
    action: np.ndarray  # (Na, 9) [t - c, R[:, 0], R[:, 1]]


def tcp_pose(p: np.ndarray, approach: np.ndarray, closing_angle: float) -> np.ndarray:
    """TCP pose whose local -z is `approach` and local x is the closing axis."""
    zax = -approach / np.linalg.norm(approach)
    ref = np.array([1.0, 0.0, 0.0]) if abs(zax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(ref, zax)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(zax, x0)
    xax = np.cos(closing_angle) * x0 + np.sin(closing_angle) * y0
    yax = np.cross(zax, xax)
    T = np.eye(4)
    T[:3, :3] = np.stack([xax, yax, zax], 1)
    T[:3, 3] = p
    return T


def propose_grasps(points: np.ndarray, c: np.ndarray, n: int, rng: np.random.Generator,
                   n_table: np.ndarray | None = None, h_table: float = 0.0,
                   max_tilt_deg: float = 45.0, depth_range: tuple[float, float] = (0.0, 0.03),
                   top_band: float = 0.05, p_any: float = 0.3, min_height: float = 0.04
                   ) -> Proposal:
    """Observation-only grasp proposal around visible surface points.

    The approach direction is tilted by up to `max_tilt_deg` from straight down onto the table
    plane estimated from the depth image (`n_table`), the usual tabletop sampling scheme; the
    TCP is placed `depth_range` past the visible point along the approach. With probability
    1 - `p_any` the point is drawn from the top `top_band` metres of the visible surface (height
    measured from the fitted table plane), and the TCP is kept at least `min_height` above the
    plane so the fingertips clear the table.
    """
    if len(points) == 0:
        points = c[None]
    up = n_table if n_table is not None else np.array([0.0, -1.0, 0.0])
    down = -up
    hp = h_table + (points - c) @ up
    top = points[hp >= hp.max() - top_band]
    Ts, acts = [], []
    for _ in range(n):
        pool = points if rng.uniform() < p_any else top
        ps = pool[rng.integers(len(pool))]
        tilt = np.radians(max_tilt_deg) * np.sqrt(rng.uniform())
        perp = np.cross(down, rng.standard_normal(3))
        perp /= np.linalg.norm(perp)
        appr = np.cos(tilt) * down + np.sin(tilt) * perp
        tcp = ps + appr * rng.uniform(*depth_range)
        h = h_table + (tcp - c) @ up
        if h < min_height:
            tcp = tcp + (min_height - h) * up
        T = tcp_pose(tcp, appr, rng.uniform(0.0, np.pi))
        Ts.append(T)
        acts.append(np.concatenate([T[:3, 3] - c, T[:3, 0], T[:3, 1]]))
    return Proposal(T=np.stack(Ts), action=np.stack(acts).astype(np.float32))


def object_frame(T_cam_tcp: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """TCP poses (..., 4, 4) from the camera frame into the object frame of pose (R, t)."""
    T_co = np.eye(4)
    T_co[:3, :3] = R
    T_co[:3, 3] = t
    return np.linalg.inv(T_co) @ T_cam_tcp
