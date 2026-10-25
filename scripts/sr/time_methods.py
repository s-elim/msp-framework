"""Per-scene decision latency of each learned LM-Grasp method, batch size 1 (one scene, all 16
candidates), on the GPU and on 8 CPU threads. Median over 300 test scenes after 30 warm-up
scenes. The bottleneck scorer draws K = 32 posterior samples, as in evaluation; the ensemble is 5 direct models.

    python scripts/sr/time_methods.py --out results/sr/latency.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.models.lm_baselines import GraspPatchCNN, PoseRegressor, grasp_patches  # noqa: E402

T_SCALE = 10.0
RUNS = REPO / "results/sr/runs"


def msp_model(run: str, dev):
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=False), latent_dim=32)
    head = OutcomeHead(latent_dim=32, action_dim=9, gripper_dim=0, hidden=256)
    ck = torch.load(RUNS / run / "best.pth", map_location="cpu", weights_only=False)
    enc.load_state_dict(ck["encoder"])
    head.load_state_dict(ck["head"])
    return enc.to(dev).eval(), head.to(dev).eval()


@torch.no_grad()
def bench(fn, n_warm: int, n: int, sync) -> float:
    ts = []
    for i in range(n_warm + n):
        sync()
        t0 = time.perf_counter()
        fn(i)
        sync()
        if i >= n_warm:
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/latency.json"))
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    b = torch.load(REPO / "data/lm_grasp/test.pt", map_location="cpu", weights_only=False)
    acts = b["actions"].clone()
    acts[..., :3] *= T_SCALE
    res = {}
    devs = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    torch.set_num_threads(8)
    for dev in devs:
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        obs = b["obs"][: a.n + 30].float().to(dev)
        act = acts[: a.n + 30].to(dev)
        enc, head = msp_model("msp_b300_d32_s0", dev)
        ens = [msp_model(f"direct_s{k}", dev) for k in range(5)]
        gq = GraspPatchCNN(scalar_dim=4)
        gq.load_state_dict(torch.load(RUNS / "gqcnn_s0/gqcnn.pth", map_location="cpu"))
        gq = gq.to(dev).eval()
        pr = PoseRegressor(pretrained=False)
        pr.load_state_dict(torch.load(RUNS / "pose_s0/pose.pth", map_location="cpu"))
        pr = pr.to(dev).eval()
        T, K = b["T"].to(dev), b["K"].to(dev)
        crop, c = b["crop"].to(dev), b["c"].float().to(dev)

        def f_msp(i):
            z = enc(obs[i:i + 1]).rsample(32)
            return head.success_probs(z, act[i:i + 1]).mean(1)

        def f_direct(i):
            e, h = ens[0]
            return h.success_probs(e(obs[i:i + 1]).mu.unsqueeze(1), act[i:i + 1])

        def f_ens(i):
            return torch.stack([h.success_probs(e(obs[i:i + 1]).mu.unsqueeze(1),
                                                act[i:i + 1]) for e, h in ens]).mean(0)

        def f_gq(i):
            xp, xs = grasp_patches(obs[i:i + 1, 3:4], T[i:i + 1], K[i:i + 1], crop[i:i + 1],
                                   c[i:i + 1])
            return torch.sigmoid(gq(xp, xs))

        def f_pose(i):
            return pr(obs[i:i + 1])

        res[dev.type] = {name: bench(f, 30, a.n, sync) for name, f in
                         (("msp_k32", f_msp), ("direct", f_direct), ("ensemble5", f_ens),
                          ("gqcnn", f_gq), ("pose_regressor_only", f_pose))}
        res[dev.type]["device_name"] = (torch.cuda.get_device_name(0) if dev.type == "cuda"
                                        else "8 CPU threads")
        print(dev, res[dev.type], flush=True)
    res["note"] = ("median ms per scene, batch 1, 16 candidates; pose-then-plan adds the "
                   "planner's physics rollouts (see plan files' seconds field)")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
