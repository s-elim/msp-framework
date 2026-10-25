"""Shared style, data access and drawing primitives for the manuscript figures.

Colour semantics (Okabe-Ito based, colourblind safe), identical in every figure and table:
    teal      pose-posterior planning (this work)           TEAL, TEAL_LIGHT (cross-object)
    orange    FoundationPose point estimate + planner       ORANGE
    grey      annotated pose (privileged reference)         GREY (dashed boxes / lines)
    blue      learned outcome scorers (RGB-D CNN, ensemble) BLUE, BLUE_DARK
    purple    learned fusion with the planner (POISE)       PURPLE
Candidate grasps are coloured by a score with the `cividis` map; simulated outcomes are marked
with a check (lifted) or a cross (dropped).
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/sr"))

TEAL, TEAL_LIGHT, TEAL_DARK = "#009E73", "#7FCDB8", "#00664A"
ORANGE, GREY, GREY_LIGHT = "#E69F00", "#7F7F7F", "#C8C8C8"
BLUE, BLUE_DARK, PURPLE, VERM = "#56B4E9", "#0072B2", "#CC79A7", "#D55E00"
CMAP = "cividis"
MM = 1 / 25.4
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
TAGS = {"point": "fpose_p4", "point64": "fpose_p64", "posterior": "fposeU1_s0_p4",
        "two_view": "fposeTV_s0_p4", "annotated": "gt_p4", "gauss": "fposeGauss_s0_p4",
        "object": "fposeObj_s0_p4", "xo": "fposeXO_s0_p4"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 6.5,
    "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.dpi": 150, "legend.frameon": False, "axes.labelpad": 2,
})


def save(fig: plt.Figure, name: str, jpeg: bool = False) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    if jpeg:
        jpeg_rasters(OUT / f"{name}.pdf")
    print("wrote", OUT / f"{name}.pdf")


def jpeg_rasters(pdf: Path, quality: int = 88) -> None:
    """Re-encode the opaque RGB rasters of a figure PDF (photographs) as JPEG, keeping their
    resolution; rasters with transparency, masks or few channels are left untouched."""
    import io

    import pymupdf
    from PIL import Image

    d = pymupdf.open(pdf)
    for page in d:
        for img in page.get_images(full=True):
            xref, smask = img[0], img[1]
            pix = pymupdf.Pixmap(d, xref)
            if smask or pix.alpha or pix.n != 3:
                continue
            buf = io.BytesIO()
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
                buf, "JPEG", quality=quality)
            page.replace_image(xref, stream=buf.getvalue())
    tmp = pdf.with_suffix(".tmp.pdf")
    d.save(tmp, garbage=4, deflate=True)
    d.close()
    tmp.replace(pdf)


def panel_label(ax: plt.Axes, s: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom",
            ha="right")


# ----------------------------------------------------------------------------------- data
@cache
def split(name: str = "test") -> dict:
    return torch.load(REPO / "data/lm_grasp" / f"{name}.pt", map_location="cpu",
                      weights_only=False)


@cache
def plan(key: str, name: str = "test") -> dict:
    return torch.load(REPO / "results/sr/plan" / f"{TAGS.get(key, key)}_{name}.pt",
                      map_location="cpu", weights_only=False)


@cache
def posterior(name: str = "test", mode: str = "knn", seed: int = 0) -> dict:
    return dict(np.load(REPO / "results/sr/posterior" / f"{mode}_s{seed}_{name}.npz"))


@cache
def metrics() -> dict:
    return json.loads((REPO / "results/sr/metrics_session4.json").read_text())


def result(name: str) -> dict:
    return json.loads((REPO / "results/sr" / name).read_text())


@cache
def models_info() -> dict:
    return json.loads((REPO / "data/lm/models/models_info.json").read_text())


def pick(s: np.ndarray) -> np.ndarray:
    from evaluate_lm import pick as _pick

    return _pick(s)


@cache
def choices(key: str, name: str = "test") -> np.ndarray:
    """Selected candidate of every scene, with the evaluation's tie-breaking (full matrix)."""
    return pick(plan(key, name)["s"].numpy())


@cache
def surface(obj: int):  # noqa: ANN201
    from msp.pose import ModelSurface

    return ModelSurface.from_mesh(str(REPO / "data/lm/models" / f"obj_{obj:06d}.ply"))


# ----------------------------------------------------------------------------- geometry
def bbox_corners(obj: int) -> np.ndarray:
    """(8, 3) corners of the model's axis-aligned box, metres, model frame."""
    m = models_info()[str(obj)]
    lo = np.array([m["min_x"], m["min_y"], m["min_z"]]) / 1000.0
    hi = lo + np.array([m["size_x"], m["size_y"], m["size_z"]]) / 1000.0
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5),
             (2, 6), (3, 7)]


def project(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixels."""
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]],
                    1)


class Scene:
    """One LM-Grasp test scene: real RGB crop, depth points, poses and candidates."""

    def __init__(self, i: int, name: str = "test", pad: float = 1.15) -> None:
        b = split(name)
        self.i, self.obj, self.im = i, int(b["obj_id"][i]), int(b["im_id"][i])
        self.K = b["K"][i].double().numpy()
        self.R_gt, self.t_gt = b["R"][i].double().numpy(), b["t"][i].double().numpy()
        self.T = b["T"][i].double().numpy()
        self.y = b["succ"][i].numpy()
        self.exe = b["exec"][i].numpy().astype(bool)
        sd = REPO / "data/lm/test" / f"{self.obj:06d}"
        self.rgb = np.asarray(Image.open(sd / "rgb" / f"{self.im:06d}.png").convert("RGB"))
        self.depth = np.asarray(Image.open(sd / "depth" / f"{self.im:06d}.png"),
                                np.float64) / 1000.0
        self.mask = np.asarray(Image.open(sd / "mask_visib" / f"{self.im:06d}_000000.png")) > 0
        u0, v0, side = (float(x) for x in b["crop"][i])
        c = np.array([u0 + side / 2, v0 + side / 2])
        side *= pad
        self.box = (c[0] - side / 2, c[0] + side / 2, c[1] + side / 2, c[1] - side / 2)

    def show(self, ax: plt.Axes, dim: float = 0.0) -> None:
        img = self.rgb.astype(float) / 255
        if dim:
            img = img * (1 - dim) + dim
        ax.imshow(img, interpolation="bilinear")
        ax.set_xlim(self.box[0], self.box[1])
        ax.set_ylim(self.box[2], self.box[3])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def draw_box(self, ax: plt.Axes, R: np.ndarray, t: np.ndarray, color: str, lw: float = 1.0,
                 alpha: float = 1.0, ls: str = "-", zorder: int = 3) -> None:
        uv = project(self.K, bbox_corners(self.obj) @ R.T + t)
        for a, b in BOX_EDGES:
            ax.plot(*uv[[a, b]].T, color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder,
                    solid_capstyle="round")

    def draw_candidates(self, ax: plt.Axes, score: np.ndarray, half_w: float = 0.03,
                        lw: float = 1.3, norm: tuple[float, float] = (0.0, 1.0)) -> None:
        cmap = plt.get_cmap(CMAP)
        for j in np.argsort(score):
            p, x = self.T[j, :3, 3], self.T[j, :3, 0]
            ends = project(self.K, np.stack([p - half_w * x, p + half_w * x]))
            col = cmap((score[j] - norm[0]) / (norm[1] - norm[0]))
            ax.plot(*ends.T, color="white", lw=lw + 1.0, solid_capstyle="round", zorder=4)
            ax.plot(*ends.T, color=col, lw=lw, solid_capstyle="round", zorder=5)

    def mark_choice(self, ax: plt.Axes, j: int, color: str, half_w: float = 0.03,
                    size: float = 80) -> None:
        p, x = self.T[j, :3, 3], self.T[j, :3, 0]
        ends = project(self.K, np.stack([p - half_w * x, p, p + half_w * x]))
        ax.plot(*ends[[0, 2]].T, color=color, lw=2.6, solid_capstyle="round", zorder=6)
        ax.scatter(*ends[1], s=size, facecolors="none", edgecolors=color, lw=1.6, zorder=7)
        ok = self.y[j] > 0.5
        ax.text(*(ends[1] + [0.0, -0.065 * (self.box[1] - self.box[0])]),
                "✓" if ok else "✗", color=TEAL_DARK if ok else VERM, fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=8,
                path_effects=_halo())

    def occupancy(self, Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """(H, W) probability that the object covers each pixel under the pose samples."""
        from msp.pose.features import silhouette

        acc = np.zeros(self.depth.shape)
        for R, t in zip(Rs, ts, strict=True):
            acc += silhouette(surface(self.obj), R, t, self.K, self.depth.shape, dilate=1)
        return acc / len(Rs)

    def draw_heat(self, ax: plt.Axes, p: np.ndarray, cmap: str = "magma",
                  alpha: float = 0.85) -> None:
        """Pixels where the posterior samples disagree about the object's extent."""
        band = p * (1 - p) * 4  # 0 where all samples agree, 1 where half of them cover it
        rgba = plt.get_cmap(cmap)(0.25 + 0.75 * p)
        rgba[..., 3] = alpha * np.clip(band / 0.6, 0, 1)
        ax.imshow(rgba, interpolation="bilinear", zorder=2)

    # ------------------------------------------------------------------ point cloud view
    def cloud(self, n: int = 6000, ring: float = 1.15, seed: int = 0
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back-projected depth points around the object (camera frame), their colours and
        whether they are inside the detection mask."""
        u0, u1, v1, v0 = self.box
        cu, cv, half = (u0 + u1) / 2, (v0 + v1) / 2, ring * (u1 - u0) / 2
        H, W = self.depth.shape
        vs, us = np.mgrid[max(0, int(cv - half)):min(H, int(cv + half)),
                          max(0, int(cu - half)):min(W, int(cu + half))]
        vs, us = vs.ravel(), us.ravel()
        z = self.depth[vs, us]
        ok = z > 0
        vs, us, z = vs[ok], us[ok], z[ok]
        zc = np.median(self.depth[self.mask & (self.depth > 0)])
        near = np.abs(z - zc) < 0.12
        vs, us, z = vs[near], us[near], z[near]
        sel = np.random.default_rng(seed).choice(len(z), min(n, len(z)), replace=False)
        vs, us, z = vs[sel], us[sel], z[sel]
        P = np.stack([(us - self.K[0, 2]) * z / self.K[0, 0],
                      (vs - self.K[1, 2]) * z / self.K[1, 1], z], 1)
        return P, self.rgb[vs, us] / 255.0, self.mask[vs, us]

    def view(self, elev: float = 28.0, azim: float = -35.0) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and centre of a virtual camera orbiting the object (camera-frame input)."""
        c = self.R_gt @ np.zeros(3) + self.t_gt
        e, a = np.radians(elev), np.radians(azim)
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return Rx @ Ry, c


def to_view(P: np.ndarray, Rv: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Orthographic view coordinates (x right, y up, depth) of camera-frame points."""
    Q = (P - c) @ Rv.T
    return np.stack([Q[:, 0], -Q[:, 1], Q[:, 2]], 1)


def _halo() -> list:
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.2, foreground="white")]


def cloud_panel(ax: plt.Axes, sc: Scene, boxes: list[tuple[np.ndarray, np.ndarray, str, float,
                                                            float, str]],
                grippers: list[tuple[int, str]] = (), elev: float = 28.0,
                azim: float = -35.0) -> None:
    """3D view of the depth point cloud with pose boxes and chosen grippers."""
    P, C, on = sc.cloud()
    Rv, c = sc.view(elev, azim)
    Q = to_view(P, Rv, c)
    ctx = ~on
    ax.scatter(Q[ctx, 0], Q[ctx, 1], s=0.25, c=GREY_LIGHT, linewidths=0, rasterized=True)
    order = np.flatnonzero(on)[np.argsort(-Q[on, 2])]  # object points, far first
    ax.scatter(Q[order, 0], Q[order, 1], s=0.9, c=C[order], linewidths=0, rasterized=True)
    lim = np.abs(to_view(bbox_corners(sc.obj) @ sc.R_gt.T + sc.t_gt, Rv, c)[:, :2]).max() * 1.45
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    corners = bbox_corners(sc.obj)
    for R, t, col, lw, alpha, ls in boxes:
        V = to_view(corners @ R.T + t, Rv, c)
        for a, b in BOX_EDGES:
            ax.plot(*V[[a, b], :2].T, color=col, lw=lw, alpha=alpha, ls=ls,
                    solid_capstyle="round")
    for j, col in grippers:
        draw_gripper(ax, sc.T[j], Rv, c, col)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_gripper(ax: plt.Axes, T: np.ndarray, Rv: np.ndarray, c: np.ndarray, color: str,
                 width: float = 0.07, finger: float = 0.03, palm: float = 0.04,
                 lw: float = 2.0) -> None:
    """Parallel-jaw glyph: two fingers along the closing axis x, palm and stem along +z."""
    o, x, z = T[:3, 3], T[:3, 0], T[:3, 2]
    h = width / 2
    pts = np.stack([o - h * x - finger * z, o - h * x + finger * z, o + h * x + finger * z,
                    o + h * x - finger * z, o + finger * z, o + (finger + palm) * z])
    V = to_view(pts, Rv, c)
    for a, b in ((0, 1), (1, 2), (2, 3), (4, 5)):
        ax.plot(*V[[a, b], :2].T, color="white", lw=lw + 1.2, solid_capstyle="round")
        ax.plot(*V[[a, b], :2].T, color=color, lw=lw, solid_capstyle="round")
