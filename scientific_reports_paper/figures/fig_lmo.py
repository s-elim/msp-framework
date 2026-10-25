"""Fig. 6 and the scene panel of Fig. 1: whole LM-O scenes with every annotated object
(results/sr/lmo/lmo_eval.pt).

    $PY scientific_reports_paper/figures/fig_lmo.py

Frame choice (fixed rule, Methods): rank frames by the number of objects whose posterior
choice succeeds where the point-estimate choice fails; Fig. 6 shows the top two frames of that
ranking (at least 20 frames apart) and the frame with the most objects whose posterior choice
fails where the point estimate succeeds. Image panels show the scene region that contains every
annotated object (all objects, never a single-object crop). Fig. 1 picks its example in
fig_teaser.py.
"""

from __future__ import annotations

import json
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    BOX_EDGES,
    GREY,
    GREY_LIGHT,
    MM,
    ORANGE,
    REPO,
    TEAL,
    TEAL_DARK,
    VERM,
    _halo,
    bbox_corners,
    draw_gripper,
    pick,
    project,
    save,
    surface,
    to_view,
)
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

LMO = REPO / "data/lmo/test/000002"


@cache
def rows() -> list[dict]:
    return torch.load(REPO / "results/sr/lmo/lmo_eval.pt", weights_only=False)


@cache
def picks() -> dict[str, np.ndarray]:
    """Selected candidates with the evaluation's tie-breaking (full stacked matrices)."""
    return {k: pick(np.stack([r[k] for r in rows()])) for k in ("s_point", "s_post", "s_annot")}


def outcomes() -> tuple[np.ndarray, np.ndarray]:
    y = np.stack([r["succ"] for r in rows()])
    rr = np.arange(len(y))
    return y[rr, picks()["s_point"]], y[rr, picks()["s_post"]]


def choose_frames() -> list[int]:
    ok_p, ok_u = outcomes()
    im = np.array([r["im"] for r in rows()])
    frames = np.unique(im)
    wins = np.array([((ok_u > 0.5) & (ok_p < 0.5) & (im == f)).sum() for f in frames])
    loss = np.array([((ok_u < 0.5) & (ok_p > 0.5) & (im == f)).sum() for f in frames])
    chosen: list[int] = []
    for f in frames[np.lexsort((frames, -wins))]:
        if all(abs(int(f) - c) >= 20 for c in chosen):
            chosen.append(int(f))
        if len(chosen) == 2:
            break
    worst = int(frames[np.lexsort((frames, -loss))][0])
    return chosen + ([worst] if worst not in chosen else [])


class Frame:
    def __init__(self, im: int) -> None:
        self.im = im
        self.idx = [n for n, r in enumerate(rows()) if r["im"] == im]
        cam = json.loads((LMO / "scene_camera.json").read_text())[str(im)]
        self.K = np.array(cam["cam_K"], np.float64).reshape(3, 3)
        self.rgb = np.asarray(Image.open(LMO / "rgb" / f"{im:06d}.png").convert("RGB"))
        self.depth = np.asarray(Image.open(LMO / "depth" / f"{im:06d}.png"), np.float64) / 1000
        self.masks = {n: np.asarray(Image.open(LMO / "mask_visib" /
                                               f"{im:06d}_{rows()[n]['inst']:06d}.png")) > 0
                      for n in self.idx}
        uv = np.concatenate([project(self.K, bbox_corners(rows()[n]["obj"]) @ rows()[n]["R"].T
                                     + rows()[n]["t"]) for n in self.idx])
        H, W = self.depth.shape
        lo, hi = uv.min(0), uv.max(0)
        mid = (lo + hi) / 2
        half_w = max((hi - lo)[0] / 2, (hi - lo)[1] / 2 * 4 / 3) * 1.18
        half_w = min(half_w, W / 2)
        half_h = half_w * 3 / 4
        cx = np.clip(mid[0], half_w, W - half_w)
        cy = np.clip(mid[1], half_h, H - half_h)
        self.roi = (cx - half_w, cx + half_w, cy + half_h, cy - half_h)

    def _crop(self, a: np.ndarray, step: int = 1
              ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """The ROI part of an (H, W, ...) image, subsampled by `step`, and its imshow extent."""
        u0, u1 = int(np.floor(self.roi[0])), int(np.ceil(self.roi[1]))
        v0, v1 = int(np.floor(self.roi[3])), int(np.ceil(self.roi[2]))
        return a[v0:v1:step, u0:u1:step], (u0 - 0.5, u1 - 0.5, v1 - 0.5, v0 - 0.5)

    def show(self, ax: plt.Axes, dim: float = 0.0) -> None:
        img, ext = self._crop(self.rgb / 255.0, step=2 if dim else 1)  # dimmed = background
        ax.imshow(img * (1 - dim) + dim, interpolation="bilinear", extent=ext)
        ax.set_xlim(self.roi[0], self.roi[1])
        ax.set_ylim(self.roi[2], self.roi[3])
        ax.axis("off")

    def segments(self, obj: int, R: np.ndarray, t: np.ndarray) -> list[np.ndarray]:
        uv = project(self.K, bbox_corners(obj) @ R.T + t)
        return [uv[[a, b]] for a, b in BOX_EDGES]

    def box(self, ax: plt.Axes, obj: int, R: np.ndarray, t: np.ndarray, col: str, lw: float,
            alpha: float = 1.0, ls: str | tuple = "-") -> None:
        ax.add_collection(LineCollection(self.segments(obj, R, t), colors=col, linewidths=lw,
                                         alpha=alpha, linestyles=[ls], capstyle="round"))

    def boxes(self, ax: plt.Axes) -> None:
        """Posterior-disagreement heat, posterior samples, annotation and FoundationPose box."""
        from msp.pose.features import silhouette

        band = np.zeros(self.depth.shape)
        for n in self.idx:
            r = rows()[n]
            p = np.mean([silhouette(surface(r["obj"]), R.astype(float), t.astype(float), self.K,
                                    self.depth.shape, dilate=1)
                         for R, t in zip(r["Rs"], r["ts"], strict=True)], 0)
            band = np.maximum(band, p * (1 - p) * 4)
        rgba = plt.get_cmap("magma")(0.35 + 0.6 * band)
        rgba[..., 3] = 0.9 * np.clip(band / 0.6, 0, 1)
        img, ext = self._crop(rgba, step=2)
        ax.imshow(img, interpolation="bilinear", extent=ext)
        samples = [seg for n in self.idx for R, t in zip(rows()[n]["Rs"], rows()[n]["ts"],
                                                          strict=True)
                   for seg in self.segments(rows()[n]["obj"], R.astype(float), t.astype(float))]
        ax.add_collection(LineCollection(samples, colors=TEAL, linewidths=0.4, alpha=0.45))
        for n in self.idx:
            r = rows()[n]
            self.box(ax, r["obj"], r["R"], r["t"], "white", 0.8, 1.0, (0, (2.5, 1.5)))
            self.box(ax, r["obj"], r["R_fp"], r["t_fp"], ORANGE, 1.0)

    def grasp(self, ax: plt.Axes, n: int, key: str, col: str, offset: tuple[float, float],
              half_w: float = 0.03) -> None:
        r = rows()[n]
        j = int(picks()[key][n])
        T = r["T"][j].astype(float)
        p, x = T[:3, 3], T[:3, 0]
        e = project(self.K, np.stack([p - half_w * x, p, p + half_w * x]))
        ax.plot(*e[[0, 2]].T, color="white", lw=3.6, solid_capstyle="round")
        ax.plot(*e[[0, 2]].T, color=col, lw=2.3, solid_capstyle="round")
        ok = r["succ"][j] > 0.5
        ax.text(*(e[1] + np.asarray(offset)), "✓" if ok else "✗",
                color=TEAL_DARK if ok else VERM, fontsize=7.5, fontweight="bold", ha="center",
                va="center", path_effects=_halo())

    def grasps(self, ax: plt.Axes, keys: tuple[str, ...]) -> None:
        s = (self.roi[1] - self.roi[0]) / 30
        for n in self.idx:
            for key in keys:
                col = ORANGE if key == "s_point" else TEAL
                off = (-s, -s) if (key == "s_point" and len(keys) > 1) else (s, -s)
                self.grasp(ax, n, key, col, off)

    def cloud(self, ax: plt.Axes, elev: float = 35.0, azim: float = -30.0,
              n: int = 40000) -> None:
        """Depth point cloud (objects in colour, context grey) with every pose box and both
        planners' grippers, seen from a virtual camera orbiting the scene."""
        centres = np.stack([rows()[i]["t"] for i in self.idx])
        c = centres.mean(0)
        reach = np.linalg.norm(centres - c, axis=1).max() + 0.12
        vs, us = np.nonzero(self.depth > 0)
        z = self.depth[vs, us]
        P = np.stack([(us - self.K[0, 2]) * z / self.K[0, 0],
                      (vs - self.K[1, 2]) * z / self.K[1, 1], z], 1)
        near = np.linalg.norm(P - c, axis=1) < reach
        vs, us, P = vs[near], us[near], P[near]
        sel = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False)
        vs, us, P = vs[sel], us[sel], P[sel]
        on = np.zeros(self.depth.shape, bool)
        for m in self.masks.values():
            on |= m
        e, a = np.radians(elev), np.radians(azim)
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rv = Rx @ Ry
        Q = to_view(P, Rv, c)
        obj = on[vs, us]
        ax.scatter(Q[~obj, 0], Q[~obj, 1], s=0.12, c=GREY_LIGHT, linewidths=0, rasterized=True)
        o = np.flatnonzero(obj)[np.argsort(-Q[obj, 2])]
        ax.scatter(Q[o, 0], Q[o, 1], s=0.6, c=self.rgb[vs[o], us[o]] / 255, linewidths=0,
                   rasterized=True)
        corners, samp, segs = [], [], {GREY: [], ORANGE: []}
        for i in self.idx:
            r = rows()[i]
            for R, t in zip(r["Rs"], r["ts"], strict=True):
                V = to_view(bbox_corners(r["obj"]) @ R.astype(float).T + t.astype(float), Rv, c)
                samp += [V[[a_, b_], :2] for a_, b_ in BOX_EDGES]
            for R, t, col in ((r["R"], r["t"], GREY), (r["R_fp"], r["t_fp"], ORANGE)):
                V = to_view(bbox_corners(r["obj"]) @ R.T + t, Rv, c)
                corners.append(V)
                segs[col] += [V[[a_, b_], :2] for a_, b_ in BOX_EDGES]
        ax.add_collection(LineCollection(samp, colors=TEAL, linewidths=0.3, alpha=0.35))
        ax.add_collection(LineCollection(segs[GREY], colors=GREY, linewidths=0.6,
                                         linestyles=[(0, (2.5, 1.5))]))
        ax.add_collection(LineCollection(segs[ORANGE], colors=ORANGE, linewidths=0.8))
        for i in self.idx:
            for key, col in (("s_point", ORANGE), ("s_post", TEAL)):
                draw_gripper(ax, rows()[i]["T"][int(picks()[key][i])].astype(float), Rv, c, col,
                             width=0.045, finger=0.018, palm=0.022, lw=1.1)
        V = np.concatenate(corners)[:, :2]
        lo, hi = V.min(0) - 0.03, V.max(0) + 0.03
        mid = (lo + hi) / 2
        half = max((hi - lo)[0] / 2, (hi - lo)[1] / 2 * 4 / 3)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half * 3 / 4, mid[1] + half * 3 / 4)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")


def _legend(fig: plt.Figure, y: float) -> None:
    h = [Line2D([], [], color=ORANGE, lw=1.3), Line2D([], [], color=TEAL, lw=1.0),
         Line2D([], [], color=GREY, lw=0.9, ls=(0, (2.5, 1.5)))]
    fig.legend(h, ["FoundationPose estimate / point-estimate grasp",
                   "pose posterior / posterior grasp", "annotated pose"], ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, y), handlelength=1.8, columnspacing=1.4)


def qualitative(frames: list[int], name: str = "fig6_qualitative") -> None:
    fig = plt.figure(figsize=(180 * MM, 46 * MM * len(frames) + 6 * MM))
    gs = fig.add_gridspec(len(frames), 3, wspace=0.02, hspace=0.03, left=0, right=1,
                          bottom=6 / (46 * len(frames) + 6), top=1)
    for r_, im in enumerate(frames):
        fr = Frame(im)
        a = fig.add_subplot(gs[r_, 0])
        fr.show(a)
        fr.boxes(a)
        b = fig.add_subplot(gs[r_, 1])
        fr.cloud(b)
        c = fig.add_subplot(gs[r_, 2])
        fr.show(c, dim=0.3)
        fr.grasps(c, ("s_point", "s_post"))
    _legend(fig, 0.0)
    save(fig, name, jpeg=True)


if __name__ == "__main__":
    qualitative(choose_frames())
