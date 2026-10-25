"""Fig. 2 (method): every step of pose-posterior grasp planning on one LM-Grasp test scene.

    $PY scientific_reports_paper/figures/fig_method.py [--scene 1165]

Nothing is drawn by hand. Row 0: training frames of the object with their FoundationPose
estimate and annotation, and the object's residual bank. Rows 1-5: the scene's RGB-D crop, its
estimate, the fit features, the kNN retrieval (the 16 bank rows actually drawn, reproduced from
the sampler's seed and checked against the stored samples) and the posterior samples. Rows 6-9:
the 16 candidates, MuJoCo rollouts of the point-estimate and posterior choices (recomputed and
checked against the stored planner scores), the per-pose success matrix, the certificate and the
execution checks. Rollouts and renders are cached in results/sr/viz/method_rollouts_<scene>.npz.
"""

from __future__ import annotations

import argparse
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE_DARK,
    CMAP,
    GREY,
    GREY_LIGHT,
    MM,
    ORANGE,
    REPO,
    TEAL,
    TEAL_DARK,
    VERM,
    Scene,
    choices,
    cloud_panel,
    metrics,
    models_info,
    plan,
    posterior,
    project,
    save,
    split,
    surface,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

SCENE = 1165
W_MM, H_MM = 180, 128
K_NN, M_SAMPLES, N_SHOW = 64, 16, 160
SHADE = 0.55  # darkens object points in the cloud views (the lamp is white)
ZMAP, RMAP = "PuBu", "RdBu_r"
KIND = {  # stage family: (face, edge)
    "bank": ("#EEF6F3", TEAL), "input": ("#F2F2F2", "#BDBDBD"), "fp": ("#FDF1DC", ORANGE),
    "post": ("#E3F4EE", TEAL), "plan": ("#D9EFE8", TEAL_DARK), "dec": ("#E8F1F8", BLUE_DARK)}
Z_GROUPS = (("fit", 0, 3), ("sil", 3, 5), ("mask", 5, 7), ("view", 7, 10))
CHECKS = (("workspace", "workspace"), ("approach_tilt", "tilt"),
          ("table_clearance", "clearance"), ("pregrasp_workspace", "pre-grasp"))


# ------------------------------------------------------------------------------ data
def _data(i: int) -> dict:
    from msp.pose import PosteriorSampler, apply_residuals, fit_bank, symmetry_transforms

    sc = Scene(i, pad=1.1)
    fp, pp = plan("point"), plan("posterior")
    R_fp, t_fp = fp["R"][i].double().numpy(), fp["t"][i].double().numpy()
    Rs, ts = posterior("test")["R"][i].astype(float), posterior("test")["t"][i].astype(float)
    z = dict(np.load(REPO / "results/sr/posterior/features_fpose.npz"))
    assert (int(z["test_obj_id"][i]), int(z["test_im_id"][i])) == (sc.obj, sc.im)
    syms = {int(o): symmetry_transforms(models_info(), int(o))
            for o in np.unique(z["train_obj_id"])}
    bank = fit_bank(z["train_obj_id"], z["train_im_id"], z["train_feats"], z["train_R_hat"],
                    z["train_t_hat"], z["train_R_gt"], z["train_t_gt"], syms)
    smp = PosteriorSampler(bank, "knn", k=K_NN)
    feat = z["test_feats"][i]
    nb = smp.neighbours(sc.obj, feat, sc.im)
    # the draw of scripts/sr/pose_posterior.py::sample_split for this frame
    drawn = np.random.default_rng([0, 31337, sc.obj, sc.im]).choice(nb, M_SAMPLES,
                                                                     replace=False)
    R_chk, t_chk = apply_residuals(R_fp, t_fp, bank.dR[drawn], bank.dt[drawn])
    assert np.allclose(R_chk, Rs, atol=1e-4) and np.allclose(t_chk, ts, atol=1e-5), \
        "posterior samples not reproduced from the bank"
    loc, scale = smp._loc[sc.obj], smp._scale[sc.obj]
    cand = smp.candidates(sc.obj, sc.im)
    dist = np.linalg.norm((bank.feats[cand] - loc) / scale - (feat - loc) / scale, axis=1)
    rank = np.argsort(dist, kind="stable")
    assert np.array_equal(cand[rank[:K_NN]], nb)
    s_pose, s_post = pp["s_pose"][i].float().numpy(), pp["s"][i].numpy()
    s_point = fp["s"][i].numpy()
    j_pt, j_st = int(choices("point")[i]), int(choices("posterior")[i])
    q_hat = metrics()["plan"]["fposeU1_s0_p4"]["cert_0.1"]["q_hat"]
    return {"sc": sc, "i": i, "R_fp": R_fp, "t_fp": t_fp, "Rs": Rs, "ts": ts, "z": z,
            "bank": bank, "loc": loc, "scale": scale, "feat": feat, "drawn": drawn,
            "order": cand[rank], "dist": dist[rank], "s_pose": s_pose, "s_post": s_post,
            "s_point": s_point, "j_pt": j_pt, "j_st": j_st, "tau": 1.0 - q_hat,
            "sim": _rollouts(sc, i, j_pt, j_st, R_fp, t_fp, Rs, ts, s_pose, s_point),
            "check": _check(i, sc.T[j_st])}


def _rollouts(sc: Scene, i: int, j_pt: int, j_st: int, R_fp: np.ndarray, t_fp: np.ndarray,
              Rs: np.ndarray, ts: np.ndarray, s_pose: np.ndarray, s_point: np.ndarray) -> dict:
    """Outcomes y (2, 1 + M, P) of the two choices at the estimate and every sample, and renders
    (2, 4, S, S, 3) of the nominal-physics rollout at the estimate and three samples."""
    path = REPO / "results/sr/viz" / f"method_rollouts_{i}.npz"
    if path.exists():
        return dict(np.load(path))
    from plan_with_pose import _phys
    from sim_render import SimRenderer, side_view, sim_for

    from msp.data.linemod import object_frame

    sim = sim_for(sc.obj)
    phys = _phys(sim, 4, 0, i)  # the planner's draws for this scene
    poses = [(R_fp, t_fp), *zip(Rs, ts, strict=True)]
    y = np.zeros((2, len(poses), len(phys)))
    last = np.zeros((2, len(poses), sim.model.nq))
    for a, j in enumerate((j_pt, j_st)):
        for m, (R, t) in enumerate(poses):
            T = object_frame(sc.T[j], R, t)
            for p, th in enumerate(phys):
                trace: list[np.ndarray] = []
                y[a, m, p] = sim.rollout(T, th, trace=trace)[0]
                if p == 0:
                    last[a, m] = trace[-1]
    assert np.allclose(y[:, 1:].mean(2), s_pose[:, [j_pt, j_st]].T), "s_pose not reproduced"
    assert np.allclose(y[:, 0].mean(1), s_point[[j_pt, j_st]]), "point score not reproduced"
    fail = [m for m in range(1, len(poses)) if y[0, m, 0] == 0]
    lift = [m for m in range(1, len(poses)) if y[0, m, 0] == 1]
    cols = np.array([0, *sorted((fail[:2] + lift[:1] + fail[2:])[:3])])
    lookat, az = side_view(sim, phys[0], object_frame(sc.T[j_st], R_fp, t_fp))
    dist = 0.2 + 1.3 * models_info()[str(sc.obj)]["diameter"] / 1000.0
    rnd = SimRenderer(sim, 320)
    img = np.stack([[rnd.render(last[a, m], (ORANGE, TEAL)[a], lookat, dist, az, -26.0)
                     for m in cols] for a in range(2)])
    out = {"y": y, "cols": cols, "img": img}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def _check(i: int, T_cam_tcp: np.ndarray) -> str | None:
    """First failed execution check of the chosen grasp (None: all pass); no IK without arm."""
    from dry_run_grasp import table_frame

    from msp.grasp import Workspace, check_grasp, grasp_in_base

    b = split("test")
    T_base_cam = table_frame(b["n_table"][i].double().numpy(), float(b["h_table"][i]),
                             b["c"][i].double().numpy())
    ws = Workspace(lo=(-0.4, -0.4, 0.0), hi=(0.4, 0.4, 0.4), max_tilt_deg=45.0 + 1e-6,
                   table_z=0.0, min_tcp_height=0.02, standoff=0.10)  # as dry_run_grasp.py
    return check_grasp(grasp_in_base(T_base_cam, T_cam_tcp), ws)


def _bank_image(d: dict, rows: np.ndarray) -> np.ndarray:
    """(n, 18, 4) RGBA rows: standardised features | rotation | translation residual."""
    from msp.pose import log_so3

    b = d["bank"]
    own = np.flatnonzero(b.obj_id == d["sc"].obj)
    rv = np.degrees(log_so3(b.dR[rows]))
    dt = b.dt[rows] * 1000.0
    r_lim = np.percentile(np.abs(np.degrees(log_so3(b.dR[own]))), 90)
    t_lim = np.percentile(np.abs(b.dt[own] * 1000.0), 90)
    zc = plt.get_cmap(ZMAP)(np.clip(((b.feats[rows] - d["loc"]) / d["scale"] + 2.5) / 5, 0, 1))
    rc = plt.get_cmap(RMAP)(np.clip((rv / r_lim + 1) / 2, 0, 1))
    tc = plt.get_cmap(RMAP)(np.clip((dt / t_lim + 1) / 2, 0, 1))
    gap = np.ones((len(rows), 1, 4))
    return np.concatenate([zc, gap, rc, gap, tc], 1)


# ---------------------------------------------------------------------------- drawing
def _sq(h: float) -> float:
    """Figure-fraction width of a square inset of figure-fraction height h."""
    return h * H_MM / W_MM


def _inset(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    a = fig.add_axes([x, y, w, h])
    a.set_xticks([])
    a.set_yticks([])
    return a


def _panel(bg: plt.Axes, x: float, y: float, w: float, h: float, badge: str, title: str,
           kind: str) -> None:
    fc, ec = KIND[kind]
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                fc=fc, ec=ec, lw=0.8, transform=bg.transAxes))
    ty = y + h - 0.018
    bg.text(x + 0.010, ty, badge, color="white", fontsize=5.6, fontweight="bold", ha="center",
            va="center", transform=bg.transAxes,
            bbox=dict(boxstyle="circle,pad=0.16", fc=ec, ec="none"))
    bg.text(x + 0.021, ty, title, fontsize=6.3, fontweight="bold", va="center",
            transform=bg.transAxes)


def _arrow(bg: plt.Axes, p0: tuple[float, float], p1: tuple[float, float],
           col: str = "#4D4D4D") -> None:
    bg.add_patch(FancyArrowPatch(p0, p1, transform=bg.transAxes, arrowstyle="-|>",
                                 mutation_scale=6, lw=0.9, color=col, shrinkA=0, shrinkB=0))


def _eq(bg: plt.Axes, x: float, y: float, s: str, size: float = 5.9) -> None:
    bg.text(x, y, s, fontsize=size, ha="center", va="center", transform=bg.transAxes,
            linespacing=1.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY_LIGHT,
                                       lw=0.5))


def _chip(bg: plt.Axes, x: float, y: float, s: str, fc: str, tc: str = "white",
          ec: str = "none", ls: str = "-") -> None:
    bg.text(x, y, s, fontsize=5.3, ha="center", va="center", color=tc, transform=bg.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.6, ls=ls))


def _crop(obj: int, im: int, pad: float = 1.25) -> tuple[np.ndarray, tuple]:
    sd = REPO / "data/lm/test" / f"{obj:06d}"
    rgb = np.asarray(Image.open(sd / "rgb" / f"{im:06d}.png").convert("RGB"))
    vs, us = np.nonzero(np.asarray(Image.open(sd / "mask_visib" / f"{im:06d}_000000.png")))
    cu, cv = (us.min() + us.max()) / 2, (vs.min() + vs.max()) / 2
    h = pad * max(np.ptp(us), np.ptp(vs)) / 2
    return rgb, (cu - h, cu + h, cv + h, cv - h)


def _cube_icon(bg: plt.Axes, x: float, y: float, col: str, ls: str = "-") -> None:
    s, o = 0.010, 0.004
    for dx, dy in ((0, 0), (o, o * W_MM / H_MM)):
        bg.add_patch(Rectangle((x + dx, y + dy), s, s * W_MM / H_MM, fill=False, ec=col, lw=0.7,
                               ls=ls, transform=bg.transAxes))
    for cx, cy in ((0, 0), (s, 0), (0, s), (s, s)):
        bg.plot([x + cx, x + cx + o], [y + cy * W_MM / H_MM, y + (cy + o) * W_MM / H_MM],
                color=col, lw=0.7, ls=ls, transform=bg.transAxes)


# ---------------------------------------------------------------------- row 0: offline
def _row_bank(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    x0, y0, w, h = 0.005, 0.815, 0.65, 0.178
    _panel(bg, x0, y0, w, h, "0", "Residual bank", "bank")
    sc, z = d["sc"], d["z"]
    hh = 0.128
    for k, n in enumerate(d["drawn"][np.argsort([np.flatnonzero(d["order"] == r)[0]
                                                  for r in d["drawn"]])][:3]):
        a = _inset(fig, x0 + 0.012 + k * (_sq(hh) + 0.005), y0 + 0.012, _sq(hh), hh)
        rgb, box = _crop(sc.obj, int(d["bank"].im_id[n]))
        a.imshow(rgb, interpolation="bilinear")
        a.set_xlim(box[0], box[1])
        a.set_ylim(box[2], box[3])
        sc.draw_box(a, z["train_R_gt"][n], z["train_t_gt"][n], "#333333", lw=0.7, ls="--")
        sc.draw_box(a, z["train_R_hat"][n], z["train_t_hat"][n], ORANGE, lw=0.9)
        for s_ in a.spines.values():
            s_.set_visible(False)
    xe = x0 + 0.012 + 3 * (_sq(hh) + 0.005)
    _arrow(bg, (xe - 0.001, y0 + 0.076), (xe + 0.016, y0 + 0.076))
    _eq(bg, xe + 0.072, y0 + 0.076,
        "$\\delta R_n=\\hat R_n^{\\top}R_n^{\\star}$\n$\\delta t_n=t_n^{\\star}-\\hat t_n$\n"
        "$z_n=\\phi(D_n,M_n,\\hat R_n,\\hat t_n)$")
    _arrow(bg, (xe + 0.130, y0 + 0.076), (xe + 0.146, y0 + 0.076))
    rows = np.flatnonzero(d["bank"].obj_id == sc.obj)
    a = _inset(fig, xe + 0.150, y0 + 0.012, 0.165, 0.128)
    a.imshow(_bank_image(d, rows), aspect="auto", interpolation="nearest")
    for s_ in a.spines.values():
        s_.set_color(GREY_LIGHT)
    for lab, c in (("$z_n$", 4.5), ("$\\delta R_n$", 12), ("$\\delta t_n$", 16)):
        a.text(c, -0.03 * len(rows), lab, fontsize=5.6, ha="center", va="bottom",
               clip_on=False)
    _legend(bg)


def _legend(bg: plt.Axes) -> None:
    x, y, dy = 0.672, 0.966, 0.031
    items = (("box", ORANGE, "-", "FoundationPose estimate $(\\hat R,\\hat t)$"),
             ("box", "#333333", "--", "annotation $(R^{\\star},t^{\\star})$, training frames"),
             ("box", TEAL, "-", "posterior sample $T_m$"),
             ("jaw", ORANGE, "-", "$a^{\\mathrm{pt}}$: choice of the point-estimate planner"),
             ("jaw", TEAL, "-", "$a^{\\star}$: choice of the posterior planner"))
    for k, (kind, col, ls, lab) in enumerate(items):
        yy = y - k * dy
        if kind == "box":
            _cube_icon(bg, x, yy - 0.008, col, ls)
        else:
            bg.plot([x, x + 0.014], [yy, yy], color=col, lw=2.2, solid_capstyle="round",
                    transform=bg.transAxes)
        bg.text(x + 0.024, yy, lab, fontsize=5.5, va="center", transform=bg.transAxes)
    yy = y - 5 * dy
    bg.text(x + 0.007, yy, "✓", color=TEAL_DARK, fontsize=7, fontweight="bold", ha="center",
            va="center", transform=bg.transAxes)
    bg.text(x + 0.016, yy, "✗", color=VERM, fontsize=7, fontweight="bold", ha="center",
            va="center", transform=bg.transAxes)
    bg.text(x + 0.024, yy, "object lifted / dropped (MuJoCo)", fontsize=5.5, va="center",
            transform=bg.transAxes)


# ------------------------------------------------------------------ row 1: posterior
RB_Y, RB_H = 0.43, 0.365


def _row_posterior(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    sc = d["sc"]
    spec = ((0.005, 0.095, "1", "Input", "input"), (0.108, 0.140, "2", "FoundationPose", "fp"),
            (0.256, 0.195, "3", "Fit features", "post"),
            (0.459, 0.200, "4", "Residual retrieval", "post"),
            (0.667, 0.328, "5", "Pose posterior", "post"))
    for x, w, b, t, k in spec:
        _panel(bg, x, RB_Y, w, RB_H, b, t, k)
    for (x, w, *_), (x1, *_) in pairwise(spec):
        _arrow(bg, (x + w, 0.61), (x1, 0.61))
    # 1 input: RGB with the visible mask, depth (near = light) on the same crop
    hh = 0.118
    for yy, depth in ((0.617, False), (0.477, True)):
        a = _inset(fig, 0.005 + (0.095 - _sq(hh)) / 2, yy, _sq(hh), hh)
        if depth:
            u0, u1, v1, v0 = (round(v) for v in sc.box)
            win = sc.depth[max(v0, 0):v1, max(u0, 0):u1]
            win = win[win > 0]
            a.imshow(np.where(sc.depth > 0, sc.depth, np.nan), cmap="Greys",
                     vmin=np.percentile(win, 2), vmax=np.percentile(win, 98))
            a.set_xlim(sc.box[0], sc.box[1])
            a.set_ylim(sc.box[2], sc.box[3])
        else:
            sc.show(a)
            a.contour(sc.mask, levels=[0.5], colors=["white"], linewidths=0.6)
    # 2 FoundationPose
    hh = 0.172
    a = _inset(fig, 0.108 + (0.140 - _sq(hh)) / 2, 0.528, _sq(hh), hh)
    sc.show(a)
    sc.draw_box(a, d["R_fp"], d["t_fp"], ORANGE, lw=1.1)
    _eq(bg, 0.178, 0.472, "$(\\hat R,\\hat t)$")
    _fit_features(fig, bg, d)
    _retrieval(fig, bg, d)
    # 5 posterior samples on the image and on the depth point cloud
    hh = 0.205
    a = _inset(fig, 0.676, 0.505, _sq(hh), hh)
    sc.show(a)
    sc.draw_heat(a, sc.occupancy(d["Rs"], d["ts"]))
    for R, t in zip(d["Rs"], d["ts"], strict=True):
        sc.draw_box(a, R, t, TEAL, lw=0.45, alpha=0.55)
    sc.draw_box(a, d["R_fp"], d["t_fp"], ORANGE, lw=0.9)
    a = _inset(fig, 0.676 + _sq(hh) + 0.012, 0.505, _sq(hh), hh)
    boxes = [(R, t, TEAL, 0.45, 0.55, "-") for R, t in zip(d["Rs"], d["ts"], strict=True)]
    cloud_panel(a, sc, [*boxes, (d["R_fp"], d["t_fp"], ORANGE, 0.9, 1.0, "-")], shade=SHADE)
    _eq(bg, 0.831, 0.466,
        "$T_m=(\\hat R\\,\\delta R_{n_m},\\ \\hat t+\\delta t_{n_m}),\\quad m=1,\\dots,M$")


def _fit_features(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    from msp.pose.features import silhouette

    sc = d["sc"]
    hh = 0.19
    a = _inset(fig, 0.262, 0.522, _sq(hh), hh)
    sc.show(a, dim=0.55)
    vs, us = np.nonzero(sc.mask & (sc.depth > 0))
    zz = sc.depth[vs, us]
    P = np.stack([(us - sc.K[0, 2]) * zz / sc.K[0, 0], (vs - sc.K[1, 2]) * zz / sc.K[1, 1], zz],
                 1)
    dist = surface(sc.obj).tree.query((P - d["t_fp"]) @ d["R_fp"])[0] * 1000.0
    sel = np.random.default_rng(0).choice(len(P), min(2500, len(P)), replace=False)
    sel = sel[np.argsort(dist[sel])]
    a.scatter(us[sel], vs[sel], c=dist[sel], cmap=ZMAP, vmin=0, vmax=15, s=0.6, linewidths=0,
              rasterized=True)
    sil = silhouette(surface(sc.obj), d["R_fp"], d["t_fp"], sc.K, sc.depth.shape)
    a.contour(sil, levels=[0.5], colors=[ORANGE], linewidths=0.7)
    a.contour(sc.mask, levels=[0.5], colors=["#333333"], linewidths=0.5, linestyles="--")
    cb = _inset(fig, 0.266, 0.488, _sq(hh) - 0.008, 0.009)
    cb.imshow(np.linspace(0, 1, 64)[None], cmap=ZMAP, aspect="auto", extent=(0, 15, 0, 1))
    cb.set_xticks([0, 15], ["0", "15 mm"], fontsize=5)
    cb.tick_params(length=1.5, pad=1)
    zq = (d["feat"] - d["loc"]) / d["scale"]
    s = _inset(fig, 0.408, 0.522, 0.011, hh)
    s.imshow(zq[:, None], cmap=ZMAP, vmin=-2.5, vmax=2.5, aspect="auto")
    for lab, lo, hi in Z_GROUPS:
        s.plot([0.75, 0.75], [lo - 0.4, hi - 0.6], color=GREY, lw=0.5, clip_on=False)
        s.text(1.0, (lo + hi - 1) / 2, lab, fontsize=5, va="center", ha="left", clip_on=False)
    s.text(0, -0.9, "$z$", fontsize=5.8, ha="center", va="bottom", clip_on=False)


def _retrieval(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    y, h = 0.468, 0.268
    rk = np.array([np.flatnonzero(d["order"] == r)[0] for r in d["drawn"]])
    p = _inset(fig, 0.468, y, 0.030, h)
    p.plot(d["dist"][:N_SHOW], np.arange(N_SHOW), color=GREY, lw=0.7)
    p.axhline(K_NN - 0.5, color=TEAL_DARK, lw=0.7, ls="--")
    p.set_ylim(N_SHOW - 0.5, -0.5)
    lo, hi = d["dist"][0], d["dist"][N_SHOW - 1]
    p.set_xlim(lo - 0.1 * (hi - lo), hi + 0.05 * (hi - lo))
    p.fill_betweenx(np.arange(K_NN), lo - 0.1 * (hi - lo), d["dist"][:K_NN], color=TEAL,
                    alpha=0.25, lw=0)
    p.text(0.5, -0.02, "$\\|z_n-z\\|$", fontsize=5.2, ha="center", va="top",
           transform=p.transAxes)
    p.text(0.25, K_NN + 3, "$k$", fontsize=5.8, color=TEAL_DARK, va="top",
           transform=p.get_yaxis_transform())
    for s_ in ("top", "right", "bottom"):
        p.spines[s_].set_visible(False)
    a = _inset(fig, 0.503, y, 0.112, h)
    a.imshow(_bank_image(d, d["order"][:N_SHOW]), aspect="auto", interpolation="nearest")
    a.add_patch(Rectangle((-0.5, -0.5), 18, K_NN, fill=False, ec=TEAL_DARK, lw=0.9))
    for s_ in a.spines.values():
        s_.set_color(GREY_LIGHT)
    for lab, c in (("$z_n$", 4.5), ("$\\delta R_n$", 12), ("$\\delta t_n$", 16)):
        a.text(c, N_SHOW + 2, lab, fontsize=5.4, ha="center", va="top", clip_on=False)
    t = _inset(fig, 0.617, y, 0.012, h)
    t.scatter(np.zeros(len(rk)), rk, marker="<", s=6, color=TEAL, linewidths=0)
    t.set_ylim(N_SHOW - 0.5, -0.5)
    t.set_xlim(-0.6, 1)
    t.axis("off")
    t.text(0.0, -2, "$M$", fontsize=5.6, color=TEAL_DARK, ha="center", va="bottom")


# ------------------------------------------------------------------- row 2: planning
RC_Y, RC_H = 0.012, 0.405


def _row_planning(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    spec = ((0.005, 0.160, "6", "Grasp candidates", "plan"),
            (0.173, 0.380, "7", "Rollouts under the posterior", "plan"),
            (0.561, 0.220, "8", "Posterior score", "plan"),
            (0.789, 0.206, "9", "Decision", "dec"))
    for x, w, b, t, k in spec:
        _panel(bg, x, RC_Y, w, RC_H, b, t, k)
    for (x, w, *_), (x1, *_) in pairwise(spec):
        _arrow(bg, (x + w, 0.20), (x1, 0.20))
    sc = d["sc"]
    hh = 0.215
    a = _inset(fig, 0.005 + (0.160 - _sq(hh)) / 2, 0.11, _sq(hh), hh)
    sc.show(a, dim=0.35)
    others = [j for j in range(len(sc.T)) if j not in (d["j_pt"], d["j_st"])]
    for j, col, lw in [*((j, GREY, 0.9) for j in others), (d["j_pt"], ORANGE, 1.8),
                       (d["j_st"], TEAL, 1.8)]:
        p, x = sc.T[j, :3, 3], sc.T[j, :3, 0]
        ends = project(sc.K, np.stack([p - 0.03 * x, p + 0.03 * x]))
        a.plot(*ends.T, color="white", lw=lw + 1.0, solid_capstyle="round", zorder=4)
        a.plot(*ends.T, color=col, lw=lw, solid_capstyle="round", zorder=5)
    _rollout_grid(fig, bg, d)
    _scores(fig, bg, d)
    _decision(fig, bg, d)


def _rollout_grid(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    r = d["sim"]
    hh, gap = 0.122, 0.006
    x0 = 0.197
    for a_ in range(2):
        yy = 0.205 - a_ * (hh + 0.012)
        bg.text(0.186, yy + hh / 2, ("$a^{\\mathrm{pt}}$", "$a^{\\star}$")[a_], fontsize=6.4,
                color=(ORANGE, TEAL_DARK)[a_], ha="center", va="center", transform=bg.transAxes)
        for k, m in enumerate(r["cols"]):
            x = x0 + k * (_sq(hh) + gap)
            for off in (0.004, 0.002):  # the P physics draws behind the nominal one
                bg.add_patch(Rectangle((x + off, yy + off * W_MM / H_MM), _sq(hh), hh,
                                       fc="white", ec=GREY_LIGHT, lw=0.4,
                                       transform=bg.transAxes))
            a = _inset(fig, x, yy, _sq(hh), hh)
            a.imshow(r["img"][a_, k], interpolation="bilinear")
            for s_ in a.spines.values():
                s_.set_color(GREY_LIGHT)
                s_.set_linewidth(0.4)
            ok = r["y"][a_, m, 0] > 0.5
            a.text(0.88, 0.88, "✓" if ok else "✗", transform=a.transAxes, fontsize=8,
                   fontweight="bold", color=TEAL_DARK if ok else VERM, ha="center",
                   va="center")
            if a_ == 0:
                bg.text(x + _sq(hh) / 2, 0.345, "$\\hat T$" if m == 0 else f"$T_{{{m}}}$",
                        fontsize=6.0, ha="center", va="center", transform=bg.transAxes)
    _eq(bg, 0.363, 0.042, "$y(a;\\,T_m,\\theta_p)\\in\\{0,1\\},\\quad m\\leq M,\\ p\\leq P$")


def _scores(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    s_pose, s_post = d["s_pose"], d["s_post"]
    order = np.argsort(-s_post, kind="stable")
    col = {d["j_st"]: TEAL, d["j_pt"]: ORANGE}
    a = _inset(fig, 0.585, 0.205, 0.185, 0.150)
    a.imshow(s_pose[:, order], cmap=CMAP, vmin=0, vmax=1, aspect="auto",
             interpolation="nearest")
    for j, c in col.items():
        k = int(np.flatnonzero(order == j)[0])
        a.add_patch(Rectangle((k - 0.5, -0.5), 1, s_pose.shape[0], fill=False, ec=c, lw=1.3))
    a.set_ylabel("$m$", fontsize=6, labelpad=5, rotation=0, va="center")
    b = _inset(fig, 0.585, 0.065, 0.185, 0.125)
    b.bar(np.arange(len(order)), s_post[order], width=0.8,
          color=[col.get(int(j), GREY_LIGHT) for j in order])
    b.scatter(np.arange(len(order)), d["s_point"][order], s=5, color=ORANGE, zorder=3,
              linewidths=0)
    b.axhline(d["tau"], color=TEAL_DARK, lw=0.6, ls="--")
    b.text(len(order) - 0.3, d["tau"] + 0.04, "$\\hat\\tau$", fontsize=5.8, color=TEAL_DARK,
           ha="right", va="bottom")
    b.set_xlim(-0.5, len(order) - 0.5)
    b.set_ylim(0, 1.08)
    b.set_yticks([0, 1])
    b.set_xlabel("candidate $a$", fontsize=5.8, labelpad=1)
    b.set_ylabel("$\\bar s(a)$", fontsize=5.8, labelpad=0)
    for s_ in ("top", "right"):
        b.spines[s_].set_visible(False)


def _decision(fig: plt.Figure, bg: plt.Axes, d: dict) -> None:
    sc = d["sc"]
    hh = 0.150
    a = _inset(fig, 0.796, 0.222, _sq(hh), hh)
    sc.show(a, dim=0.3)
    sc.mark_choice(a, d["j_pt"], ORANGE, size=28)
    sc.mark_choice(a, d["j_st"], TEAL, size=28)
    x, y, dy = 0.951, 0.356, 0.031
    failed = [n for n, _ in CHECKS].index(d["check"]) if d["check"] else len(CHECKS)
    for k, (_, lab) in enumerate(CHECKS):
        ok = k < failed
        _chip(bg, x, y - k * dy, f"{lab} {'✓' if ok else '✗'}",
              "white", TEAL_DARK if ok else VERM, ec=TEAL_DARK if ok else VERM)
    _chip(bg, x, y - 4 * dy, "IK", "white", GREY, ec=GREY, ls="--")
    s = float(d["s_post"][d["j_st"]])
    _chip(bg, 0.892, 0.180, f"$\\bar s(a^{{\\star}})={s:.2f}\\geq\\hat\\tau={d['tau']:.2f}$",
          TEAL_DARK if s >= d["tau"] else GREY)
    _chip(bg, 0.892, 0.132, "↻ second view if $\\bar s\\leq\\tau_{2}$", "white",
          TEAL_DARK, ec=TEAL_DARK, ls="--")
    _eq(bg, 0.892, 0.066,
        "$T_{\\mathrm{base}}^{\\mathrm{flange}}=T_{\\mathrm{base}}^{\\mathrm{cam}}"
        "\\,T_{\\mathrm{cam}}^{\\mathrm{tcp}}\\,T_{\\mathrm{tcp}}^{\\mathrm{flange}}$")


def _connectors(bg: plt.Axes) -> None:
    _arrow(bg, (0.555, 0.815), (0.555, RB_Y + RB_H))  # bank -> retrieval
    _arrow(bg, (0.055, RB_Y), (0.055, RC_Y + RC_H))  # observation -> candidates
    yc = (RB_Y + RC_Y + RC_H) / 2  # pose samples -> rollouts, through the gap between rows
    bg.plot([0.831, 0.831, 0.40], [RB_Y, yc, yc], color="#4D4D4D", lw=0.9,
            transform=bg.transAxes, solid_capstyle="butt")
    _arrow(bg, (0.40, yc), (0.40, RC_Y + RC_H))


def draw(scene: int = SCENE, name: str = "fig2_method") -> None:
    d = _data(scene)
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    _row_bank(fig, bg, d)
    _row_posterior(fig, bg, d)
    _row_planning(fig, bg, d)
    _connectors(bg)
    save(fig, name, jpeg=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=SCENE)
    a = ap.parse_args()
    draw(a.scene)
