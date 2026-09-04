"""Qualitative panel: what MSP selects, and what the analytic proxy selects instead.

    python scripts/paper/make_qualitative_fig.py

Reads only cached artefacts, never renders and never touches a GPU:

    results/libero/corpus/libero_*.pt   the L1 test corpus (seed 4, 2000 scenes)
    results/libero/l1_scores.pt         s_msp, the analytic margin, and outcomes

Alignment between the two is asserted, not assumed: object_index, succ, margin and
executable must match element for element, otherwise the scores belong to a different
corpus and the figure would attribute grasps to the wrong scenes.

Candidate grasps are projected into view 0 with the camera the corpus was rendered
from: a ring of 8 cameras at radius 0.24 m and height 0.26 m, each looking at the
origin with fovy 48 degrees (src/msp/oracle/libero_sim.py::_cameras_xml). Nothing is
re-simulated, so every mark traces to a recorded rollout.

Panels are the scenes where the two selectors disagree and the ground-truth rollouts
resolve the disagreement in MSP's favour. That is a selected subset by construction;
the population numbers behind it are in results/libero/l1_proxy_vs_msp.json
(precision@25%: MSP 0.984, analytic 0.503).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("qualitative")

COL, DCOL = 3.5, 7.16
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 7,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Camera ring used to render the corpus.
CAM_R, CAM_H, FOVY, N_VIEWS = 0.24, 0.26, 48.0, 8

# The two jaws sit at x = -0.05 and +0.05 in the hand frame
# (src/msp/oracle/libero_sim.py, bodies finger_l / finger_r), so the closing axis
# is the hand's local x. Drawing it is what makes two grasps at the same point but
# different yaw distinguishable; a dot alone cannot show the disagreement.
JAW_HALF = 0.05

GREEN = "#1B7F4B"
RED = "#B3202C"
GREY = "#6B7280"


def camera_basis(view: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = 2 * np.pi * view / N_VIEWS
    pos = np.array([CAM_R * np.cos(theta), CAM_R * np.sin(theta), CAM_H])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def project(points: np.ndarray, view: int, size: int) -> np.ndarray:
    """World xyz -> pixel uv in the rendered view. Points behind the camera return NaN."""
    pos, fwd, right, up = camera_basis(view)
    d = np.atleast_2d(points) - pos
    depth = d @ fwd
    focal = (size / 2) / np.tan(np.deg2rad(FOVY) / 2)
    u = size / 2 + focal * (d @ right) / depth
    v = size / 2 - focal * (d @ up) / depth
    uv = np.stack([u, v], axis=-1)
    uv[depth <= 1e-6] = np.nan
    return uv


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-9)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def jaw_segment(action: np.ndarray) -> np.ndarray:
    """Endpoints of the gripper's closing axis for one 7-vector action."""
    axis = quat_to_matrix(action[3:7]) @ np.array([1.0, 0.0, 0.0])
    centre = action[:3]
    return np.stack([centre - JAW_HALF * axis, centre + JAW_HALF * axis])


def load(corpus: Path, scores: Path) -> tuple[dict, dict]:
    c = torch.load(corpus, map_location="cpu", weights_only=False, mmap=True)
    s = torch.load(scores, map_location="cpu", weights_only=False)
    checks = {
        "object_index": torch.equal(c["object_index"], s["object_index"]),
        "succ": torch.equal(c["succ"].squeeze(-1), s["succ"]),
        "margin": torch.equal(c["margin"].squeeze(-1), s["margin"]),
        "executable": torch.equal(c["executable"], s["executable"]),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"corpus and scores disagree on {bad}; wrong split?")
    return c, s


def pick_scenes(s: dict, n: int) -> list[int]:
    """Scenes where MSP's top grasp succeeds and the analytic proxy's top grasp fails."""
    msp = s["s_msp"].clone()
    proxy = s["margin"].clone()
    ok = s["executable"]
    msp[~ok] = -np.inf
    proxy[~ok] = -np.inf
    succ = s["succ"]

    i_msp = msp.argmax(dim=1)
    i_proxy = proxy.argmax(dim=1)
    rows = torch.arange(len(succ))
    disagree = (i_msp != i_proxy) & (succ[rows, i_msp] > 0.5) & (succ[rows, i_proxy] < 0.5)
    cand = torch.nonzero(disagree).squeeze(-1).tolist()

    # Spread the panels over distinct objects rather than showing one object n times.
    chosen, seen = [], set()
    for i in cand:
        o = int(s["object_index"][i])
        if o not in seen:
            chosen.append(i)
            seen.add(o)
        if len(chosen) == n:
            break
    for i in cand:                       # top up if fewer than n distinct objects matched
        if len(chosen) == n:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen


def build(c: dict, s: dict, idx: list[int], view: int, cols: int):
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DCOL, DCOL / cols * rows * 1.02))
    size = c["observation"].shape[-1]

    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        rgb = c["observation"][i, view, :3].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(rgb, 0, 1))

        acts = c["actions"][i].numpy()
        uv = project(acts[:, :3], view, size)
        ok = s["executable"][i].numpy()
        m = np.where(ok, s["s_msp"][i].numpy(), -np.inf)
        p = np.where(ok, s["margin"][i].numpy(), -np.inf)
        i_msp, i_proxy = int(m.argmax()), int(p.argmax())

        for k in np.nonzero(ok)[0]:
            seg = project(jaw_segment(acts[k]), view, size)
            ax.plot(seg[:, 0], seg[:, 1], color=GREY, lw=0.7, alpha=0.5, zorder=3)
        for k, color, lw in ((i_proxy, RED, 1.6), (i_msp, GREEN, 1.8)):
            seg = project(jaw_segment(acts[k]), view, size)
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=4,
                    solid_capstyle="round")
        ax.scatter(*uv[i_proxy], s=9, color=RED, zorder=5, linewidths=0)
        ax.scatter(*uv[i_msp], s=9, color=GREEN, zorder=6, linewidths=0)

        # Zoom on the object; at 96 px the grasp geometry is otherwise 10 px wide.
        cu, cv = np.nanmean(uv[ok], axis=0)
        half = 22
        # Clamp the window inside the frame, or a scene near the edge renders
        # with a blank margin where the image runs out.
        cu = float(np.clip(cu, half, size - half))
        cv = float(np.clip(cv, half, size - half))
        ax.set_xlim(cu - half, cu + half)
        ax.set_ylim(cv + half, cv - half)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#C8D0DC")
            sp.set_linewidth(0.6)

    for ax in np.atleast_1d(axes).ravel()[len(idx):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="results/libero/corpus/libero_197aaee1a663467e.pt")
    ap.add_argument("--scores", default="results/libero/l1_scores.pt")
    ap.add_argument("--out", default="paper/figures/fig_qualitative")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--view", type=int, default=0)
    args = ap.parse_args()

    c, s = load(Path(args.corpus), Path(args.scores))
    idx = pick_scenes(s, args.n)
    log.info("scenes: %s", idx)
    for i in idx:
        log.info("  scene %4d  %-22s msp_top=%.3f proxy_top=%.3f",
                 i, s["object_names"][int(s["object_index"][i])],
                 float(s["s_msp"][i].max()), float(s["margin"][i].max()))

    fig = build(c, s, idx, args.view, args.cols)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    log.info("wrote %s.pdf and %s.png", out, out)


if __name__ == "__main__":
    main()
