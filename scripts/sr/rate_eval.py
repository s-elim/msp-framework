"""Payload size versus decision quality for trained MSP runs.

The belief (mu, log sigma^2) is uniformly quantized per dimension to b bits on a range fixed
from the CALIBRATION split (1st-99th percentile). The code length is the cross-entropy of the
test symbols under per-dimension symbol frequencies estimated on the calibration split
(add-one smoothing), i.e. what a static entropy coder fitted before deployment would spend.
Decisions are recomputed from the dequantized belief with the trained head, and the
certificate is recalibrated on dequantized calibration beliefs.

    python scripts/sr/rate_eval.py --runs 'msp_b*_d32_s*' --out results/sr/rate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/sr"))

from evaluate_lm import certificate, ranking_metrics  # noqa: E402
from msp.belief import DiagonalGaussianBelief  # noqa: E402
from msp.models import OutcomeHead  # noqa: E402

BITS = (1, 2, 3, 4, 6, 8)
T_SCALE = 10.0


def quantize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, bits: int):
    L = 2 ** bits
    span = (hi - lo).clamp_min(1e-6)
    idx = ((x - lo) / span * (L - 1)).round().clamp(0, L - 1)
    return lo + idx / (L - 1) * span, idx.long(), L


def code_bits(sym_cal: torch.Tensor, sym_te: torch.Tensor, L: int) -> float:
    tot = 0.0
    for j in range(sym_cal.shape[1]):
        cnt = torch.bincount(sym_cal[:, j], minlength=L).double() + 1.0
        p = cnt / cnt.sum()
        tot += float(-torch.log2(p[sym_te[:, j]]).mean())
    return tot


@torch.no_grad()
def scores(head, mu, lv, actions, k: int = 32) -> np.ndarray:
    torch.manual_seed(0)
    out = []
    for i in range(0, len(mu), 256):
        b = DiagonalGaussianBelief(mu[i:i + 256], lv[i:i + 256])
        out.append(head.success_probs(b.rsample(k), actions[i:i + 256]).mean(1))
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="msp_b*_d32_s*")
    ap.add_argument("--data", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--out", default=str(REPO / "results/sr/rate.json"))
    a = ap.parse_args()
    lab = {k: torch.load(Path(a.data) / f"{k}.pt", map_location="cpu", weights_only=False)
           for k in ("calib", "test")}
    acts = {}
    for k, v in lab.items():
        x = v["actions"].clone()
        x[..., :3] *= T_SCALE
        acts[k] = x
    y = {k: v["succ"].numpy() for k, v in lab.items()}
    res = {}
    for d in sorted((REPO / "results/sr/runs").glob(a.runs)):
        if not (d / "scores.pt").exists() or "_ho" in d.name:
            continue
        r = torch.load(d / "scores.pt", map_location="cpu", weights_only=False)
        lat = r["args"]["latent"]
        head = OutcomeHead(latent_dim=lat, action_dim=9, gripper_dim=0, hidden=256)
        ck = torch.load(d / "best.pth", map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        head.eval()
        sc = r["scores"]
        x_cal = torch.cat([sc["calib"]["mu"], sc["calib"]["logvar"]], -1)
        x_te = torch.cat([sc["test"]["mu"], sc["test"]["logvar"]], -1)
        lo = torch.quantile(x_cal, 0.01, dim=0)
        hi = torch.quantile(x_cal, 0.99, dim=0)
        rows = []
        s_te = scores(head, sc["test"]["mu"], sc["test"]["logvar"], acts["test"])
        s_ca = scores(head, sc["calib"]["mu"], sc["calib"]["logvar"], acts["calib"])
        m = ranking_metrics(s_te, y["test"], lab["test"]["obj_id"].numpy())
        c = certificate(s_ca, y["calib"], s_te, y["test"], 0.1)
        rows.append({"bits_per_dim": 32, "payload_bits": 2 * lat * 32.0,
                     "within_scene_auc": m["within_scene_auc"], "top1_success": m["top1_success"],
                     "cert_coverage": c["coverage"], "cert_fraction": c["certified_fraction"]})
        for b in BITS:
            qc, symc, L = quantize(x_cal, lo, hi, b)
            qt, symt, _ = quantize(x_te, lo, hi, b)
            s_te = scores(head, qt[:, :lat], qt[:, lat:], acts["test"])
            s_ca = scores(head, qc[:, :lat], qc[:, lat:], acts["calib"])
            m = ranking_metrics(s_te, y["test"], lab["test"]["obj_id"].numpy())
            c = certificate(s_ca, y["calib"], s_te, y["test"], 0.1)
            rows.append({"bits_per_dim": b, "payload_bits": code_bits(symc, symt, L),
                         "fixed_width_bits": 2 * lat * b,
                         "within_scene_auc": m["within_scene_auc"],
                         "top1_success": m["top1_success"], "cert_coverage": c["coverage"],
                         "cert_fraction": c["certified_fraction"]})
            print(d.name, b, round(rows[-1]["payload_bits"], 1),
                  round(m["within_scene_auc"], 3), round(m["top1_success"], 3), flush=True)
        res[d.name] = {"beta": r["args"]["beta"], "latent": lat, "seed": r["args"]["seed"],
                       "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
