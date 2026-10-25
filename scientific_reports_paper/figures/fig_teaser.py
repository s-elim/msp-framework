"""Fig. 1 (teaser): an accurate pose estimate, a dropped object, and the fix at a glance.

    $PY scientific_reports_paper/figures/fig_teaser.py

a, a whole LM-O test frame with the FoundationPose estimate of every annotated object. b, one
object of it (fixed rule, `choose_instance`): its estimate passes ADD(-S), yet the grasp
planned on it drops the object, while planning on the pose posterior lifts it. The rollouts
are MuJoCo at the annotated pose with the instance's hidden physics, recomputed and checked
against the stored outcomes (cache: results/sr/viz/teaser_rollouts_<n>.npz). c, the main
results.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from common import (
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
    metrics,
    models_info,
    project,
    result,
    save,
    surface,
)
from fig_lmo import LMO, Frame, choose_frames, outcomes, picks, rows
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image

W_MM, H_MM = 180, 62
ADD_PASS = 0.10  # ADD(-S) threshold, fraction of the object diameter


def choose_instance() -> tuple[int, float]:
    """The LM-O instance outside the Fig. 6 frames with the largest visible area among those
    where the FoundationPose estimate passes ADD(-S), the point-estimate planner rates its
    choice 1 and that choice fails, and the posterior rates that choice below the certificate
    threshold while certifying its own choice, which succeeds.

    Returns (row index, ADD(-S) error / diameter)."""
    from evaluate_lm import add_metrics, model_points

    rs = rows()
    ok_p, ok_u = outcomes()
    tau = 1.0 - metrics()["plan"]["fposeU1_s0_p4"]["cert_0.1"]["q_hat"]
    rr = np.arange(len(rs))
    s_pt, s_po = np.stack([r["s_point"] for r in rs]), np.stack([r["s_post"] for r in rs])
    jp, js = picks()["s_point"], picks()["s_post"]
    flagged = (s_pt[rr, jp] == 1.0) & (s_po[rr, jp] < tau) & (s_po[rr, js] >= tau)
    obj = np.array([r["obj"] for r in rs])
    im = np.array([r["im"] for r in rs])
    diam = {int(k): v["diameter"] / 1000.0 for k, v in models_info().items()}
    err = np.asarray(add_metrics(np.stack([r["R_fp"] for r in rs]),
                                 np.stack([r["t_fp"] for r in rs]),
                                 np.stack([r["R"] for r in rs]), np.stack([r["t"] for r in rs]),
                                 obj, model_points(REPO / "data/lm"), diam)["add_err_m"])
    rel = err / np.array([diam[int(o)] for o in obj])
    keep = np.flatnonzero((rel < ADD_PASS) & (ok_p < 0.5) & (ok_u > 0.5) & flagged
                          & ~np.isin(im, choose_frames()))
    names = [f"{rs[n]['im']:06d}_{rs[n]['inst']:06d}.png" for n in keep]
    area = [int((np.asarray(Image.open(LMO / "mask_visib" / f)) > 0).sum()) for f in names]
    n = int(keep[int(np.argmax(area))])
    return n, float(rel[n])


def instance_sim(n: int) -> tuple:
    """(simulator, hidden physics, {"s_point", "s_post"}: TCP pose of that planner's choice
    in the object frame) of LM-O instance n. The observation, the 16 proposals and the scene
    physics are rebuilt exactly as scripts/sr/lmo_eval.py::_instance builds them, and checked
    against the stored proposals and outcomes."""
    from sim_render import sim_for

    from msp.data.linemod import object_frame, propose_grasps, table_plane

    r = rows()[n]
    o, im, k = int(r["obj"]), int(r["im"]), int(r["inst"])
    info = json.loads((LMO / "scene_gt_info.json").read_text())[str(im)][k]
    K = Frame(im).K
    dep = np.asarray(Image.open(LMO / "depth" / f"{im:06d}.png"), np.float32) / 1000.0
    msk = np.asarray(Image.open(LMO / "mask_visib" / f"{im:06d}_{k:06d}.png")) > 0
    vs, us = np.nonzero(msk & (dep > 0))
    z = dep[vs, us].astype(np.float64)
    pts = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
    c = pts.mean(0)
    n_tab, h_tab = table_plane(dep, msk, tuple(int(v) for v in info["bbox_visib"]), K, c,
                               np.random.default_rng([o, im, 3]))
    rng = np.random.default_rng([0, o, im])
    prop = propose_grasps(pts, c, 16, rng, n_table=n_tab, h_table=float(h_tab))
    assert np.allclose(prop.T, r["T"], atol=1e-5), "proposals not reproduced"
    sim = sim_for(o)
    phys = sim.sample_phys(rng)
    grasp = {}
    for key in ("s_point", "s_post"):
        j = int(picks()[key][n])
        grasp[key] = object_frame(prop.T[j], r["R"], r["t"])
        assert sim.rollout(grasp[key], phys)[0] == r["succ"][j], "outcome not reproduced"
    return sim, phys, grasp


def _rollouts(n: int) -> dict:
    """Final states of the two choices at the annotated pose under the instance's hidden
    physics, rendered side on: img (2, S, S, 3) for (point choice, posterior choice)."""
    path = REPO / "results/sr/viz" / f"teaser_rollouts_{n}.npz"
    if path.exists():
        return dict(np.load(path))
    from sim_render import SimRenderer, side_view

    sim, phys, grasp = instance_sim(n)
    last = []
    for key in ("s_point", "s_post"):
        trace: list[np.ndarray] = []
        sim.rollout(grasp[key], phys, trace=trace)
        last.append(trace[-1])
    lookat, az = side_view(sim, phys, grasp["s_post"])
    dist = 0.2 + 1.3 * models_info()[str(rows()[n]["obj"])]["diameter"] / 1000.0
    rnd = SimRenderer(sim, 400)
    img = np.stack([rnd.render(q, col, lookat, dist, az, -26.0)
                    for q, col in zip(last, (ORANGE, TEAL), strict=True)])
    out = {"img": img}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def _zoom(fr: Frame, n: int, pad: float = 1.12) -> tuple[float, float, float, float]:
    r = rows()[n]
    uv = project(fr.K, bbox_corners(r["obj"]) @ r["R"].T + r["t"])
    c, h = (uv.min(0) + uv.max(0)) / 2, pad * (uv.max(0) - uv.min(0)).max() / 2
    return c[0] - h, c[0] + h, c[1] + h, c[1] - h


def _jaw(ax: plt.Axes, fr: Frame, n: int, key: str, col: str) -> None:
    T = rows()[n]["T"][int(picks()[key][n])].astype(float)
    p, x = T[:3, 3], T[:3, 0]
    e = project(fr.K, np.stack([p - 0.03 * x, p + 0.03 * x]))
    ax.plot(*e.T, color="white", lw=4.0, solid_capstyle="round", zorder=6)
    ax.plot(*e.T, color=col, lw=2.6, solid_capstyle="round", zorder=7)


def _mm(fig: plt.Figure) -> tuple[float, float]:
    return fig.get_figwidth() / MM, fig.get_figheight() / MM


def _tile(fig: plt.Figure, x: float, y: float, s: float, col: str) -> plt.Axes:
    w, h = _mm(fig)
    a = fig.add_axes([x / w, y / h, s / w, s / h])
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True)
        sp.set_color(col)
        sp.set_linewidth(1.2)
    return a


def _story(fig: plt.Figure, fr: Frame, n: int, rel: float, x0: float, s: float) -> None:
    """Two rows (point estimate, posterior) of [zoomed image, MuJoCo outcome]."""
    from msp.pose.features import silhouette

    r = rows()[n]
    zoom = _zoom(fr, n)
    imgs = _rollouts(n)["img"]
    for row, (key, col) in enumerate((("s_point", ORANGE), ("s_post", TEAL))):
        y = 3.5 + (1 - row) * (s + 3.5)
        a = _tile(fig, x0, y, s, col)
        fr.show(a)
        if key == "s_point":
            fr.box(a, r["obj"], r["R"], r["t"], "white", 0.9, 1.0, (0, (2.5, 1.5)))
            fr.box(a, r["obj"], r["R_fp"], r["t_fp"], ORANGE, 1.4)
            a.text(0.04, 0.05, f"ADD error {100 * rel:.0f}% of $d$", transform=a.transAxes,
                   fontsize=5.6, color="white", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.25", fc=ORANGE, ec="none"))
        else:
            p = np.mean([silhouette(surface(r["obj"]), R.astype(float), t.astype(float), fr.K,
                                    fr.depth.shape, dilate=1)
                         for R, t in zip(r["Rs"], r["ts"], strict=True)], 0)
            band = p * (1 - p) * 4
            rgba = plt.get_cmap("magma")(0.35 + 0.6 * band)
            rgba[..., 3] = 0.85 * np.clip(band / 0.6, 0, 1)
            a.imshow(rgba, interpolation="bilinear")
            segs = [seg for R, t in zip(r["Rs"], r["ts"], strict=True)
                    for seg in fr.segments(r["obj"], R.astype(float), t.astype(float))]
            a.add_collection(LineCollection(segs, colors=TEAL, linewidths=0.6, alpha=0.6))
        _jaw(a, fr, n, key, col)
        a.set_xlim(zoom[0], zoom[1])
        a.set_ylim(zoom[2], zoom[3])
        a.axis("on")
        fig.text((x0 - 2.0) / _mm(fig)[0], (y + s / 2) / _mm(fig)[1],
                 ("point estimate", "posterior (ours)")[row], color=(ORANGE, TEAL_DARK)[row],
                 rotation=90, ha="center", va="center", fontsize=6.2, fontweight="bold")
        b = _tile(fig, x0 + s + 2.0, y, s, col)
        b.imshow(imgs[row], interpolation="bilinear")
        ok = r["succ"][int(picks()[key][n])] > 0.5
        b.text(0.86, 0.86, "✓" if ok else "✗", transform=b.transAxes, fontsize=13,
               fontweight="bold", color=TEAL_DARK if ok else VERM, ha="center", va="center",
               path_effects=_halo())


def _dumbbell(ax: plt.Axes) -> None:
    m = metrics()["plan"]
    pa, lmo = result("posterior_analysis.json"), result("lmo/lmo_metrics.json")["all"]
    b5 = next(b for b in pa["bins"] if b["lo"] == 0.05)
    cert = {k: m[t]["cert_0.1"]["certified_precision"]
            for k, t in (("p", "fpose_p4"), ("u", "fposeU1_s0_p4"), ("a", "gt_p4"))}
    rows_ = (("LM-Grasp", m["fpose_p4"]["top1_success"], m["fposeU1_s0_p4"]["top1_success"],
              m["gt_p4"]["top1_success"]),
             ("error 5-10% $d$", b5["point"], b5["posterior"], b5["annotated"]),
             ("LM-O, occluded", lmo["s_point"]["top1"], lmo["s_post"]["top1"],
              lmo["s_annot"]["top1"]),
             ("certified grasps", cert["p"], cert["u"], cert["a"]))
    for k, (_, p, u, a) in enumerate(rows_):
        y = len(rows_) - 1 - k
        if k % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F4F4F4", lw=0, zorder=0)
        ax.annotate("", (u, y), (p, y), arrowprops=dict(arrowstyle="-|>", color=GREY_LIGHT,
                                                        lw=1.6, shrinkA=3, shrinkB=4))
        ax.scatter([a], [y], s=22, marker="D", facecolors="white", edgecolors=GREY,
                   linewidths=0.9, zorder=2)
        ax.scatter([p], [y], s=26, color=ORANGE, zorder=3, linewidths=0)
        ax.scatter([u], [y], s=34, color=TEAL, zorder=4, linewidths=0)
        ax.text(p - 0.008, y + 0.16, f"{p:.2f}", fontsize=5.4, color="#9A6A00", ha="right",
                va="bottom")
        ax.text(u + 0.008, y + 0.16, f"{u:.2f}", fontsize=5.4, color=TEAL_DARK, ha="left",
                va="bottom", fontweight="bold")
    ax.set_yticks(range(len(rows_)), [r[0] for r in rows_][::-1], fontsize=6)
    ax.set_ylim(-0.6, len(rows_) - 0.4)
    ax.set_xlim(0.64, 1.02)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("grasp success", fontsize=6, labelpad=1)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=4),
         plt.Line2D([], [], marker="o", ls="", color=TEAL, ms=4.5),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=GREY, ms=3.6, mew=0.9)]
    ax.legend(h, ["point estimate", "posterior (ours)", "annotated pose"], fontsize=5.4,
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, handlelength=1.2,
              columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)


def teaser(name: str = "fig1_teaser") -> None:
    n, rel = choose_instance()
    r = rows()[n]
    print("teaser instance", n, "frame", r["im"], "object", r["obj"], f"ADD/d {rel:.3f}")
    fr = Frame(int(r["im"]))
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    wa = 70.0
    a = fig.add_axes([1.0 / W_MM, 3.5 / H_MM, wa / W_MM, 0.75 * wa / H_MM])
    fr.show(a)
    for m in fr.idx:
        q = rows()[m]
        fr.box(a, q["obj"], q["R_fp"], q["t_fp"], ORANGE, 1.1 if m == n else 0.8,
               1.0 if m == n else 0.85)
    z = _zoom(fr, n)
    a.add_patch(Rectangle((z[0], z[3]), z[1] - z[0], z[2] - z[3], fill=False, ec="white",
                          lw=1.2, ls=(0, (3, 1.5))))
    s = 25.5
    _story(fig, fr, n, rel, wa + 6.0, s)
    c = fig.add_axes([(wa + 6.0 + 2 * s + 2.0 + 21.0) / W_MM, 8.5 / H_MM,
                      (W_MM - (wa + 6.0 + 2 * s + 2.0 + 21.0) - 2.0) / W_MM, 44.0 / H_MM])
    _dumbbell(c)
    for x, lab in ((1.0, "a"), (wa + 2.0, "b"), (wa + 6.0 + 2 * s + 5.0, "c")):
        fig.text(x / W_MM, 0.985, lab, fontsize=8, fontweight="bold", va="top")
    save(fig, name, jpeg=True)


def story(path: str, s: float = 25.5) -> None:
    """Panel b alone (the zoomed object and its two rollouts), for the project page."""
    n, rel = choose_instance()
    fig = plt.figure(figsize=((2 * s + 8.5) * MM, (2 * s + 10.5) * MM))
    _story(fig, Frame(int(rows()[n]["im"])), n, rel, 4.5, s)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    teaser()
