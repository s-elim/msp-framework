"""Check that LM model +z is the table normal, the assumption `linemod_sim` rests on.

For sampled real test images, fit a plane to the depth pixels in a ring around the object (the
board it stands on) and compare its normal with the model's z axis in the camera frame.

    python scripts/sr/check_lm_upright.py --out results/sr/lm_upright.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

LM13 = (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)


def plane_normal(pts: np.ndarray, rng: np.random.Generator, iters: int = 200,
                 tol: float = 0.004) -> np.ndarray | None:
    best, best_n = 0, None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < tol
        if inl.sum() > best:
            best, best_n = int(inl.sum()), n
    return best_n if best > 0.3 * len(pts) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lm")
    ap.add_argument("--per-object", type=int, default=20)
    ap.add_argument("--out", default="results/sr/lm_upright.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(a.root) / "test"
    rows = {}
    for oid in LM13:
        sd = root / f"{oid:06d}"
        gt = json.loads((sd / "scene_gt.json").read_text())
        cam = json.loads((sd / "scene_camera.json").read_text())
        info = json.loads((sd / "scene_gt_info.json").read_text())
        keys = rng.choice(sorted(gt, key=int), a.per_object, replace=False)
        angs = []
        for k in keys:
            R = np.array(gt[k][0]["cam_R_m2c"]).reshape(3, 3)
            K = np.array(cam[k]["cam_K"]).reshape(3, 3)
            dep = np.asarray(Image.open(sd / "depth" / f"{int(k):06d}.png"), dtype=np.float32)
            x, y, w, h = info[k][0]["bbox_obj"]
            pad = int(0.6 * max(w, h))
            v0, v1 = max(0, y - pad), min(dep.shape[0], y + h + pad)
            u0, u1 = max(0, x - pad), min(dep.shape[1], x + w + pad)
            mask = np.zeros_like(dep, bool)
            mask[v0:v1, u0:u1] = True
            mask[y:y + h, x:x + w] = False
            mask &= dep > 0
            vs, us = np.nonzero(mask)
            if len(vs) < 200:
                continue
            sel = rng.choice(len(vs), min(3000, len(vs)), replace=False)
            z = dep[vs[sel], us[sel]] / 1000.0
            pts = np.stack([(us[sel] - K[0, 2]) * z / K[0, 0], (vs[sel] - K[1, 2]) * z / K[1, 1],
                            z], 1)
            n = plane_normal(pts, rng)
            if n is None:
                continue
            angs.append(float(np.degrees(np.arccos(min(1.0, abs(n @ R[:, 2]))))))
        rows[oid] = {"n": len(angs), "median_deg": float(np.median(angs)) if angs else None,
                     "p90_deg": float(np.percentile(angs, 90)) if angs else None}
        print(oid, rows[oid], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
