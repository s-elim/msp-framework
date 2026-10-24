"""Train one method on LM-Grasp and write its scores for the val, calib and test splits.

    python scripts/sr/train_lm.py --method msp --beta 300 --latent 32 --seed 0
    python scripts/sr/train_lm.py --method direct --seed 0      # same net, z = mu, no rate
    python scripts/sr/train_lm.py --method pose --seed 0        # pose regressor (then plan_with_pose)
    python scripts/sr/train_lm.py --method gqcnn --seed 0       # grasp-aligned patch scorer
    ... --holdout 2,9,12,15   # unseen-object protocol: these objects never enter training

Every method sees the same crops, the same candidate grasps and the same labels, and trains
for the same number of epochs on the same CPU budget. Output: results/sr/runs/<tag>/scores.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.belief import DiagonalGaussianBelief  # noqa: E402
from msp.data.synthetic import collate  # noqa: E402
from msp.engine.trainer import TrainConfig, Trainer  # noqa: E402
from msp.math.bottleneck import BetaSchedule  # noqa: E402
from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.models.lm_baselines import GraspPatchCNN, PoseRegressor, grasp_patches  # noqa: E402
from msp.utils.seed import seed_everything  # noqa: E402

log = logging.getLogger("train_lm")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SCALE = 10.0  # action translation in decimetres, so all 9 action coordinates are O(1)


class LMGrasp(Dataset):
    def __init__(self, path: Path, keep: torch.Tensor | None = None) -> None:
        b = torch.load(path, map_location="cpu", weights_only=False)
        if keep is not None:
            b = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v)
                 for k, v in b.items()}
        self.b = b
        a = b["actions"].clone()
        a[..., :3] *= T_SCALE
        self.actions = a
        self.n = b["obs"].shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        b = self.b
        na = self.actions.shape[1]
        return {"observation": b["obs"][i].float(), "actions": self.actions[i],
                "weights": torch.ones(na) / na, "succ": b["succ"][i][:, None],
                "margin": torch.zeros(na, 1), "slip": b["slip"][i][:, None],
                "index": torch.tensor(i)}


def load_split(name: str, data: Path, holdout: set[int], unseen: bool) -> LMGrasp:
    b = torch.load(data / f"{name}.pt", map_location="cpu", weights_only=False)
    if not holdout:
        return LMGrasp(data / f"{name}.pt")
    is_held = torch.tensor([int(o) in holdout for o in b["obj_id"]])
    keep = is_held if unseen else ~is_held
    return LMGrasp(data / f"{name}.pt", keep=keep)


@torch.no_grad()
def score_split(enc, head, ds, direct: bool, k: int = 32) -> dict:
    was = enc.training
    enc.eval()
    head.eval()
    torch.manual_seed(1234)
    mus, lvs, ss, vs = [], [], [], []
    for i in range(0, len(ds), 128):
        obs = ds.b["obs"][i:i + 128].float().to(DEV)
        act = ds.actions[i:i + 128].to(DEV)
        bel = enc(obs)
        z = bel.mu.unsqueeze(1) if direct else bel.rsample(k)
        p = head.success_probs(z, act)  # (B, K, Na)
        mus.append(bel.mu.cpu())
        lvs.append(bel.logvar.cpu())
        ss.append(p.mean(1).cpu())
        vs.append(p.var(1, unbiased=False).cpu())
    enc.train(was)
    head.train(was)
    return {"s": torch.cat(ss), "v": torch.cat(vs), "mu": torch.cat(mus),
            "logvar": torch.cat(lvs)}


def bce_of(s: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), y.float()))


# ----------------------------------------------------------------------------- MSP / direct
def train_msp(a, dsets, out: Path, direct: bool) -> dict:
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=a.latent)
    head = OutcomeHead(latent_dim=a.latent, action_dim=9, gripper_dim=0, hidden=256)
    dl = {k: DataLoader(v, batch_size=a.batch, shuffle=(k == "train"), collate_fn=collate,
                        num_workers=0, drop_last=(k == "train"))
          for k, v in dsets.items() if k in ("train", "val")}
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=1e-4, warmup_epochs=2,
                      amp_dtype="off", beta=BetaSchedule.sufficiency_for_success(a.beta),
                      out_dir=str(out), seed=a.seed, train_samples=8,
                      kl_warmup_epochs=15,
                      deterministic_warmup_epochs=(a.epochs + 1) if direct else 10)
    tr = Trainer(enc, head, dl["train"], dl["val"], cfg, DEV)
    hist = []
    best, best_state = math.inf, None
    while tr.epoch < cfg.epochs:
        t0 = time.time()
        m = tr.train_epoch() | tr.validate()
        # Model selection, identical in kind for every method: the validation success BCE of
        # the method's OWN decision score (MSP: posterior-mean probability over K = 8 draws;
        # direct: probability at z = mu).
        sv = score_split(enc, head, dsets["val"], direct, k=8)["s"]
        bce = bce_of(sv, dsets["val"].b["succ"])
        m["val/select_bce"] = bce
        if bce < best:
            best = bce
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().clone() for k, v in head.state_dict().items()})
            tr.save("best.pth")
        m["epoch"], m["sec"] = tr.epoch, time.time() - t0
        hist.append(m)
        log.info("ep %d %.0fs train %.4f val %.4f rate %.3f sel-bce %.4f", tr.epoch, m["sec"],
                 m.get("loss/total", float("nan")), m.get("val/loss/total", float("nan")),
                 m.get("val/loss/rate_kl", float("nan")), bce)
        tr.epoch += 1
    tr.save("last.pth")
    enc.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    scores = {split: score_split(enc, head, dsets[split], direct, k=32)
              for split in ("val", "calib", "test")}
    log.info("selected epoch %d (val BCE %.4f)",
             int(np.argmin([h["val/select_bce"] for h in hist])), best)
    return {"scores": scores, "history": hist}


# ----------------------------------------------------------------------------- pose
_SYM: dict[int, torch.Tensor] = {}


def symmetries(root: Path) -> dict[int, torch.Tensor]:
    info = json.loads((root / "models" / "models_info.json").read_text())
    out = {}
    for k, v in info.items():
        mats = [torch.eye(3)]
        for s in v.get("symmetries_discrete", []):
            mats.append(torch.tensor(s, dtype=torch.float32).reshape(4, 4)[:3, :3])
        out[int(k)] = torch.stack(mats)
    return out


def pose_loss(Rh, dth, R, dt, obj, syms) -> torch.Tensor:
    losses = []
    for i in range(Rh.shape[0]):
        S = syms[int(obj[i])]  # (M, 3, 3); R S is an equivalent pose for a symmetric object
        cand = R[i].unsqueeze(0) @ S
        lr = ((Rh[i].unsqueeze(0) - cand) ** 2).sum((1, 2)).min()
        losses.append(lr)
    return torch.stack(losses).mean() + 10.0 * (dth - dt).abs().sum(-1).mean()


def train_pose(a, dsets, out: Path) -> dict:
    syms = symmetries(REPO / "data/lm")
    net = PoseRegressor(pretrained=True).to(DEV)
    syms = {k: v.to(DEV) for k, v in syms.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    tr = dsets["train"]
    n = len(tr)
    hist = []
    best, best_state = math.inf, None
    for ep in range(a.epochs):
        lr = a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * ep / a.epochs)))
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n - a.batch + 1, a.batch):
            j = perm[i:i + a.batch]
            obs = tr.b["obs"][j].float().to(DEV)
            R = tr.b["R"][j].to(DEV)
            dt = (tr.b["t"][j] - tr.b["c"][j]).to(DEV)
            Rh, dth = net(obs)
            loss = pose_loss(Rh, dth, R, dt, tr.b["obj_id"][j], syms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(j)
        net.eval()
        with torch.no_grad():
            va = dsets["val"]
            vl = []
            for i in range(0, len(va), 128):
                Rh, dth = net(va.b["obs"][i:i + 128].float().to(DEV))
                vl.append(float(pose_loss(Rh, dth, va.b["R"][i:i + 128].to(DEV),
                                          (va.b["t"][i:i + 128] - va.b["c"][i:i + 128]).to(DEV),
                                          va.b["obj_id"][i:i + 128], syms)) * len(Rh))
            vloss = sum(vl) / len(va)
        if vloss < best:
            best, best_state = vloss, {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist.append({"epoch": ep, "loss": tot / n, "val_loss": vloss, "sec": time.time() - t0})
        log.info("pose ep %d loss %.4f val %.4f %.0fs", ep, tot / n, vloss, time.time() - t0)
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), out / "pose.pth")
    net.eval()
    scores = {}
    with torch.no_grad():
        for split in ("val", "calib", "test"):
            ds = dsets[split]
            Rs, ts = [], []
            for i in range(0, len(ds), 128):
                Rh, dth = net(ds.b["obs"][i:i + 128].float().to(DEV))
                Rs.append(Rh.cpu())
                ts.append(ds.b["c"][i:i + 128] + dth.cpu())
            scores[split] = {"R_hat": torch.cat(Rs), "t_hat": torch.cat(ts)}
    return {"scores": scores, "history": hist}


# ----------------------------------------------------------------------------- GQ-CNN
def _patches(ds, idx):
    depth = ds.b["obs"][idx, 3:4].float().to(DEV)
    return grasp_patches(depth, ds.b["T"][idx].to(DEV), ds.b["K"][idx].to(DEV),
                         ds.b["crop"][idx].to(DEV), ds.b["c"][idx].float().to(DEV))


def train_gqcnn(a, dsets, out: Path) -> dict:
    net = GraspPatchCNN(scalar_dim=4).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tr = dsets["train"]
    n = len(tr)
    bs = max(8, a.batch // 2)
    hist = []
    best, best_state = math.inf, None
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            xp, xs = _patches(tr, j)
            y = tr.b["succ"][j].reshape(-1).to(DEV)
            loss = F.binary_cross_entropy_with_logits(net(xp, xs), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sv = gq_scores(net, dsets["val"])
        vb = bce_of(sv, dsets["val"].b["succ"])
        if vb < best:
            best, best_state = vb, {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist.append({"epoch": ep, "loss": tot / n, "val_bce": vb, "sec": time.time() - t0})
        log.info("gqcnn ep %d loss %.4f val %.4f %.0fs", ep, tot / n, vb, time.time() - t0)
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), out / "gqcnn.pth")
    net.eval()
    scores = {split: {"s": gq_scores(net, dsets[split])} for split in ("val", "calib", "test")}
    return {"scores": scores, "history": hist}


@torch.no_grad()
def gq_scores(net, ds) -> torch.Tensor:
    was = net.training
    net.eval()
    ss = []
    for i in range(0, len(ds), 64):
        j = torch.arange(i, min(i + 64, len(ds)))
        xp, xs = _patches(ds, j)
        ss.append(torch.sigmoid(net(xp, xs)).reshape(len(j), -1).cpu())
    net.train(was)
    return torch.cat(ss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["msp", "direct", "pose", "gqcnn"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=300.0)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--holdout", default="", help="comma-separated obj ids held out of training")
    ap.add_argument("--data", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    seed_everything(a.seed)
    os.environ.setdefault("TORCH_HOME", "/nas/Selim/selim_sarowar/.cache/torch")

    holdout = {int(x) for x in a.holdout.split(",") if x}
    data = Path(a.data)
    dsets = {"train": load_split("train", data, holdout, unseen=False),
             "val": load_split("val", data, holdout, unseen=False),
             "calib": load_split("calib", data, holdout, unseen=bool(holdout)),
             "test": load_split("test", data, holdout, unseen=bool(holdout))}
    tag = a.tag or (f"{a.method}" + (f"_b{a.beta:g}_d{a.latent}" if a.method == "msp" else "")
                    + (f"_ho{a.holdout.replace(',', '-')}" if holdout else "") + f"_s{a.seed}")
    out = REPO / "results/sr/runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s: train %d val %d calib %d test %d", tag,
             *(len(dsets[k]) for k in ("train", "val", "calib", "test")))
    t0 = time.time()
    if a.method in ("msp", "direct"):
        res = train_msp(a, dsets, out, direct=(a.method == "direct"))
    elif a.method == "pose":
        res = train_pose(a, dsets, out)
    else:
        res = train_gqcnn(a, dsets, out)
    res["args"] = vars(a)
    res["wall_seconds"] = time.time() - t0
    res["device"] = str(DEV)
    res["split_index"] = {k: {"obj_id": v.b["obj_id"], "im_id": v.b["im_id"]}
                          for k, v in dsets.items() if k != "train"}
    torch.save(res, out / "scores.pt")
    (out / "done.json").write_text(json.dumps({"wall_seconds": res["wall_seconds"],
                                               "args": vars(a)}, indent=1))
    log.info("done %s in %.0fs", tag, res["wall_seconds"])


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
