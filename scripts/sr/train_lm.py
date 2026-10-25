"""Train one method on LM-Grasp and write its scores for the val, calib and test splits.

    python scripts/sr/train_lm.py --method msp --beta 300 --latent 32 --seed 0
    python scripts/sr/train_lm.py --method direct --seed 0      # same net, z = mu, no rate
    python scripts/sr/train_lm.py --method pose --seed 0        # pose regressor (then plan_with_pose)
    python scripts/sr/train_lm.py --method gqcnn --seed 0       # grasp-aligned patch scorer
    ... --holdout 2,9,12,15   # unseen-object protocol: these objects never enter training

Every method sees the same crops, the same candidate grasps and the same labels, trains for the
same number of epochs with the same training-time augmentation (`augment`), and is selected by
its own validation loss among epochs >= SELECT_FROM. Output: results/sr/runs/<tag>/scores.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.belief import DiagonalGaussianBelief  # noqa: E402
from msp.data.collate import collate  # noqa: E402
from msp.engine.trainer import TrainConfig, Trainer  # noqa: E402
from msp.math.bottleneck import BetaSchedule, distortion, vib_objective  # noqa: E402
from msp.types import Outcome  # noqa: E402
from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.models.lm_baselines import GraspPatchCNN, PoseRegressor, grasp_patches  # noqa: E402
from msp.models.points import PointBeliefEncoder  # noqa: E402
from msp.utils.seed import seed_everything  # noqa: E402

log = logging.getLogger("train_lm")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SCALE = 10.0  # action translation in decimetres, so all 9 action coordinates are O(1)


SELECT_FROM = 5  # checkpoints are selected among epochs >= 5 for every method


def augment(obs: torch.Tensor, g: torch.Generator | None = None) -> torch.Tensor:
    """Photometric and depth noise applied identically to every image-based method's training
    crops: per-image brightness/contrast (+-20 %), per-channel gain (+-5 %), depth noise
    (sd 0.01 in the normalised depth units, i.e. 2.5 mm). (B, 4, S, S) or (4, S, S)."""
    one = obs.dim() == 3
    x = obs.unsqueeze(0) if one else obs.clone()
    b = x.shape[0]
    dev = x.device
    br = 1 + 0.4 * (torch.rand(b, 1, 1, 1, generator=g) - 0.5).to(dev)
    gain = 1 + 0.1 * (torch.rand(b, 3, 1, 1, generator=g) - 0.5).to(dev)
    rgb = x[:, :3]
    m = rgb.mean(dim=(1, 2, 3), keepdim=True)
    x[:, :3] = ((rgb - m) * br + m) * gain
    x[:, :3] = x[:, :3].clamp(0, 1)
    valid = x[:, 3:4] != 0
    x[:, 3:4] = torch.where(valid, x[:, 3:4] + 0.01 * torch.randn(x[:, 3:4].shape,
                                                                   generator=g).to(dev),
                            x[:, 3:4])
    return x[0] if one else x


PLAN = REPO / "results/sr/plan"
N_CUR_4D = 1024  # current-frame points in the 4D cloud; the two context frames add 512 each


def load_points(path: Path, b: dict, keep: torch.Tensor | None, mode: str) -> torch.Tensor:
    """Study 6 point clouds (N, 2048, 7) float16 [x, y, z (dm, camera frame, minus c), t, rgb]
    from data/lm_grasp/points_<split>.pt (scripts/sr/build_points.py).

    3d: the 2048 current-frame points, t = 0. 4d: 1024 current-frame points plus the two
    registered context frames (512 each, t = frame offset / 10); a missing context frame is
    replaced by further current-frame points (t = 0), so every scene has the same budget."""
    P = torch.load(path.parent / f"points_{path.stem}.pt", map_location="cpu",
                   weights_only=False)
    oid = P["obj_id"] if keep is None else P["obj_id"][keep]
    iid = P["im_id"] if keep is None else P["im_id"][keep]
    assert torch.equal(oid, b["obj_id"]) and torch.equal(iid, b["im_id"]), \
        f"{path.stem}: point clouds are not aligned with the corpus rows"
    cur, ctx = P["cur"].float(), P["ctx"].float()
    dt, ok = P["ctx_dt"].float(), P["ctx_ok"]
    if keep is not None:
        cur, ctx, dt, ok = cur[keep], ctx[keep], dt[keep], ok[keep]
    zeros = torch.zeros((*cur.shape[:2], 1))
    if mode == "3d":
        pts = torch.cat([cur[..., :3] * T_SCALE, zeros, cur[..., 3:]], -1)
    elif mode == "4d":
        parts = [torch.cat([cur[:, :N_CUR_4D, :3] * T_SCALE, zeros[:, :N_CUR_4D],
                            cur[:, :N_CUR_4D, 3:]], -1)]
        for k in range(2):
            fill = cur[:, N_CUR_4D + 512 * k: N_CUR_4D + 512 * (k + 1)]
            src = torch.where(ok[:, k, None, None], ctx[:, k], fill)
            t = torch.where(ok[:, k], dt[:, k] / 10.0, torch.zeros_like(dt[:, k]))
            parts.append(torch.cat([src[..., :3] * T_SCALE,
                                    t[:, None, None].expand(-1, src.shape[1], 1),
                                    src[..., 3:]], -1))
        pts = torch.cat(parts, 1)
    else:
        raise ValueError(f"unknown points mode {mode!r}")
    return pts.half()


class LMGrasp(Dataset):
    points_mode = ""  # study 6: "", "3d" or "4d"; set once in main() from --points

    def __init__(self, path: Path, keep: torch.Tensor | None = None, train: bool = False,
                 teacher: torch.Tensor | None = None, alpha: float = 0.0,
                 planner: str = "", aux: torch.Tensor | None = None) -> None:
        self.train = train
        self.teacher, self.alpha = teacher, alpha
        # study 3 V1b: per-candidate VLM score in [0, 1] as an auxiliary regression target on
        # the head's margin channel (otherwise a zero target at weight 0.01)
        self.aux = aux[keep] if (aux is not None and keep is not None) else aux
        if teacher is not None and keep is not None:
            self.teacher = teacher[keep]
        b = torch.load(path, map_location="cpu", weights_only=False)
        if keep is not None:
            b = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v)
                 for k, v in b.items()}
        self.b = b
        a = b["actions"].clone()
        a[..., :3] *= T_SCALE
        if planner:  # C4: each candidate's pose-then-plan score as a 10th action feature
            ps = torch.load(PLAN / f"{planner}_p4_{path.stem}.pt", map_location="cpu",
                            weights_only=False)["s"].float()
            if keep is not None:
                ps = ps[keep]
            a = torch.cat([a, (2 * ps - 1)[..., None]], -1)
        self.actions = a
        self.n = b["obs"].shape[0]
        self.pts = load_points(path, b, keep, self.points_mode) if self.points_mode else None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        b = self.b
        na = self.actions.shape[1]
        obs = b["obs"][i].float()
        if self.train:
            obs = augment(obs)
        y = b["succ"][i].float()
        soft = y if self.teacher is None else (1 - self.alpha) * y + self.alpha * self.teacher[i]
        margin = torch.zeros(na, 1) if self.aux is None else self.aux[i][:, None]
        item = {"observation": obs, "actions": self.actions[i],
                "weights": torch.ones(na) / na, "succ": soft[:, None], "succ_hard": y,
                "margin": margin, "slip": b["slip"][i][:, None],
                "index": torch.tensor(i)}
        if self.pts is not None:
            p = self.pts[i].float()
            if self.train:  # 2 mm point jitter, the point analogue of `augment`'s depth noise
                p[:, :3] += 0.02 * torch.randn(p.shape[0], 3)
            item["points"] = p
        return item


TEACHER = REPO / "results/sr/teacher/train_direct_s5-14.pt"


def make_teacher() -> None:
    """Mean success prediction on the TRAIN split of the 10 direct scorers with seeds 5-14
    (redesign candidate C2). These seeds are never evaluated as single models."""
    ds = LMGrasp(REPO / "data/lm_grasp/train.pt")
    ps = []
    for k in range(5, 15):
        enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=False), latent_dim=32)
        head = OutcomeHead(latent_dim=32, action_dim=9, gripper_dim=0, hidden=256)
        ck = torch.load(REPO / f"results/sr/runs/direct_s{k}/best.pth", map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
        head.load_state_dict(ck["head"])
        ps.append(score_split(enc.to(DEV), head.to(DEV), ds, direct=True)["s"])
    TEACHER.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"t": torch.stack(ps).mean(0), "members": list(range(5, 15))}, TEACHER)
    log.info("teacher written: %s", TEACHER)


VLM = REPO / "results/sr/vlm"


def vlm_targets(name: str, a) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Study 3 teacher tensors (N, Na) in [0, 1] for split `name`, or (None, None).

    --vlm-labels: the VLM agreement score / 10 (scenes whose reply failed to parse fall back to
    the hard label for distillation and to the split mean for the auxiliary target).
    --vlm-control const: t = 0.5 everywhere (label smoothing). shuffle: VLM scores permuted
    across scenes of the same object (seeded), which keeps their marginal statistics and object
    base rates but breaks the scene-specific geometry."""
    if not a.vlm_labels:
        return None, None
    y = torch.load(Path(a.data) / f"{name}.pt", map_location="cpu", weights_only=False)
    L = torch.load(VLM / f"labels_{a.vlm_labels}_{name}_v2.pt", map_location="cpu",
                   weights_only=False)
    t = L["ratings"].float() / 10.0
    miss = torch.isnan(t).any(1)
    if a.vlm_control == "shuffle":
        g = torch.Generator().manual_seed(1000 + a.seed)
        oid = y["obj_id"]
        t2 = t.clone()
        for o in oid.unique():
            ix = torch.nonzero((oid == o) & ~miss).squeeze(1)
            t2[ix] = t[ix[torch.randperm(len(ix), generator=g)]]
        t = t2
    elif a.vlm_control == "const":
        t = torch.full_like(t, 0.5)
    if getattr(a, "vlm_mix_ens", False):  # V2: mean of the VLM and the C2 ensemble teacher
        ens = torch.load(TEACHER, map_location="cpu", weights_only=False)["t"].float()
        t = torch.where(miss[:, None], ens, 0.5 * (t + ens))
        miss = torch.zeros_like(miss)
    teach = torch.where(miss[:, None], y["succ"].float(), t)
    aux = torch.where(miss[:, None], t[~miss].mean(), t)
    return (teach if a.distill > 0 else None), (aux if a.vlm_aux > 0 else None)


def load_split(name: str, data: Path, holdout: set[int], unseen: bool,
               distill: float = 0.0, planner: str = "", a=None) -> LMGrasp:
    train = name == "train"
    if train and a is not None and a.vlm_labels:
        teach, aux = vlm_targets(name, a)
        keep = None
        if holdout:
            b = torch.load(data / f"{name}.pt", map_location="cpu", weights_only=False)
            keep = ~torch.tensor([int(o) in holdout for o in b["obj_id"]])
        return LMGrasp(data / f"{name}.pt", keep=keep, train=True, teacher=teach,
                       alpha=distill, planner=planner, aux=aux)
    if not holdout:
        if train and distill > 0:
            t = torch.load(TEACHER, map_location="cpu", weights_only=False)["t"]
            return LMGrasp(data / f"{name}.pt", train=True, teacher=t, alpha=distill,
                           planner=planner)
        return LMGrasp(data / f"{name}.pt", train=train, planner=planner)
    b = torch.load(data / f"{name}.pt", map_location="cpu", weights_only=False)
    is_held = torch.tensor([int(o) in holdout for o in b["obj_id"]])
    keep = is_held if unseen else ~is_held
    return LMGrasp(data / f"{name}.pt", keep=keep, train=train, planner=planner)


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
        if ds.pts is not None:  # study 6: RGB-D crop + point cloud
            pts = ds.pts[i:i + 128].float().to(DEV)
            bel = enc(obs, pts)
            act = enc.grasp_features(pts, act)
        else:
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


def pairwise_rank_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Within-scene pairwise logistic loss. logit, y: (B, Na). Mean over (success, failure)
    pairs of the same scene; scenes with one outcome only contribute nothing."""
    d = logit[:, :, None] - logit[:, None, :]  # (B, Na, Na): l_i - l_j
    m = (y[:, :, None] > 0.5) & (y[:, None, :] < 0.5)
    if not m.any():
        return logit.sum() * 0.0
    return F.softplus(-d[m]).mean()


class LMTrainer(Trainer):
    """Trainer with an optional within-scene ranking term (redesign candidate C1), added on the
    marginal success logit log(p_bar / (1 - p_bar)), p_bar = mean over the K posterior draws."""

    rank_w: float = 0.0

    def _forward(self, batch):
        pts = batch.get("points")
        ranked = self.rank_w > 0 and self.encoder.training
        if pts is None and not ranked:
            return super()._forward(batch)
        # one forward pass shared by the VIB terms and the ranking term (parent logic inlined;
        # amp is off for every LM-Grasp run, so the parent's autocast is a no-op)
        obs = batch["observation"].to(self.device)
        act = batch["actions"].to(self.device)
        y = Outcome(succ=batch["succ"].to(self.device), margin=batch["margin"].to(self.device),
                    slip=batch["slip"].to(self.device))
        if pts is not None:  # study 6: RGB-D crop + point cloud, local features per candidate
            pts = pts.to(self.device)
            bel = self.encoder(obs, pts)
            act = self.encoder.grasp_features(pts, act)
        else:
            bel = self.encoder(obs)
        z = bel.mu.unsqueeze(1) if self._deterministic_now() else bel.rsample(
            self.cfg.train_samples)
        pred = self.head(z, act)
        terms = vib_objective(pred.float(), y.expand_to(z.shape[1]), bel.mu.float(),
                              bel.logvar.float(), self.cfg.beta,
                              weights=batch["weights"].to(self.device),
                              rate_weight=self._rate_weight_at(self.epoch))
        if not ranked:
            return terms
        p = torch.sigmoid(pred.succ_logit.float()).squeeze(-1).mean(1).clamp(1e-6, 1 - 1e-6)
        rl = pairwise_rank_loss(torch.log(p) - torch.log1p(-p),
                                batch["succ_hard"].to(self.device))
        return replace(terms, loss=terms.loss + self.cfg.beta.succ * self.rank_w * rl)


# ------------------------------------------------------------ bottleneck scorer (msp) / direct
def train_msp(a, dsets, out: Path, direct: bool) -> dict:
    act_dim = dsets["train"].actions.shape[-1]
    if a.points:  # study 6
        enc = PointBeliefEncoder(latent_dim=a.latent)
        act_dim += enc.local.output_dim
    else:
        enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True),
                            latent_dim=a.latent)
    head = OutcomeHead(latent_dim=a.latent, action_dim=act_dim, gripper_dim=0, hidden=256)
    dl = {k: DataLoader(v, batch_size=a.batch, shuffle=(k == "train"), collate_fn=collate,
                        num_workers=0, drop_last=(k == "train"))
          for k, v in dsets.items() if k in ("train", "val")}
    beta = BetaSchedule.success_first(a.beta)
    if a.vlm_aux > 0:  # study 3 V1b: VLM score regressed on the margin channel
        beta = replace(beta, margin=a.vlm_aux * a.beta)
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=1e-4, warmup_epochs=2,
                      amp_dtype="off", beta=beta,
                      out_dir=str(out), seed=a.seed, train_samples=8,
                      kl_warmup_epochs=3,
                      deterministic_warmup_epochs=(a.epochs + 1) if direct else 2)
    tr = LMTrainer(enc, head, dl["train"], dl["val"], cfg, DEV)
    tr.rank_w = a.rank
    hist = []
    best, best_state = math.inf, None
    while tr.epoch < cfg.epochs:
        t0 = time.time()
        m = tr.train_epoch() | tr.validate()
        # Model selection, identical in kind for every method: the validation success BCE of
        # the method's OWN decision score (bottleneck scorer: posterior-mean probability over K = 8 draws;
        # direct: probability at z = mu).
        sv = score_split(enc, head, dsets["val"], direct, k=8)["s"]
        bce = bce_of(sv, dsets["val"].b["succ"])
        m["val/select_bce"] = bce
        if bce < best and tr.epoch >= SELECT_FROM:
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
    enc.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    scores = {split: score_split(enc, head, dsets[split], direct, k=32) for split in a.splits}
    log.info("selected epoch %d (val BCE %.4f)",
             SELECT_FROM + int(np.argmin([h["val/select_bce"] for h in hist][SELECT_FROM:])), best)
    return {"scores": scores, "history": hist}


# ----------------------------------------------------------------------------- C3: qcode
class FactorizedPrior(torch.nn.Module):
    """Static per-dimension Gaussian entropy model for an integer-quantised code. The code length
    of y (B, d) is -sum_j log2 P(y_j), P = Gaussian mass of the unit bin around y_j (Balle et
    al. 2017, without the hyperprior)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros(d))
        self.log_s = torch.nn.Parameter(torch.zeros(d))

    def bits(self, y: torch.Tensor) -> torch.Tensor:
        s = self.log_s.exp().clamp_min(0.05)
        n = torch.distributions.Normal(self.m, s)
        p = (n.cdf(y + 0.5) - n.cdf(y - 0.5)).clamp_min(1e-9)
        return -(torch.log2(p)).sum(-1)


@torch.no_grad()
def qcode_scores(enc, head, prior, ds) -> dict:
    was = enc.training
    enc.eval()
    head.eval()
    ss, bs, cs = [], [], []
    for i in range(0, len(ds), 128):
        y = torch.round(enc(ds.b["obs"][i:i + 128].float().to(DEV)).mu)
        act = ds.actions[i:i + 128].to(DEV)
        ss.append(head.success_probs(y.unsqueeze(1), act).squeeze(1).cpu())
        bs.append(prior.bits(y).cpu())
        cs.append(y.cpu().to(torch.int16))
    enc.train(was)
    head.train(was)
    return {"s": torch.cat(ss), "bits": torch.cat(bs), "code": torch.cat(cs)}


def train_qcode(a, dsets, out: Path) -> dict:
    """Redesign candidate C3: deterministic code, additive uniform noise as the quantisation
    proxy in training, rounding at test, loss = beta-weighted outcome NLL + lam * code bits.
    Same encoder, head, data, augmentation, epochs, optimiser and selection rule as `direct`."""
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=a.latent)
    head = OutcomeHead(latent_dim=a.latent, action_dim=9, gripper_dim=0, hidden=256)
    prior = FactorizedPrior(a.latent)
    enc, head, prior = enc.to(DEV), head.to(DEV), prior.to(DEV)
    beta = BetaSchedule.success_first(a.beta)
    params = list(enc.parameters()) + list(head.parameters()) + list(prior.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(dsets["train"], batch_size=a.batch, shuffle=True, collate_fn=collate,
                    num_workers=0, drop_last=True)
    hist, best, best_state = [], math.inf, None
    for ep in range(a.epochs):
        w = 2
        lr = a.lr * ((ep + 1) / w if ep < w else
                     0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (ep - w) / max(1, a.epochs - w))))
        for g in opt.param_groups:
            g["lr"] = lr
        lam = 0.0 if ep < 2 else a.lam * min(1.0, (ep - 1) / 3.0)
        enc.train()
        head.train()
        t0, tot, tb, n = time.time(), 0.0, 0.0, 0
        for batch in dl:
            obs = batch["observation"].to(DEV)
            y = enc(obs).mu
            yt = y + (torch.rand_like(y) - 0.5)
            pred = head(yt.unsqueeze(1), batch["actions"].to(DEV))
            tgt = Outcome(succ=batch["succ"].to(DEV), margin=batch["margin"].to(DEV),
                          slip=batch["slip"].to(DEV)).expand_to(1)
            bits = prior.bits(yt).mean()
            loss = distortion(pred, tgt).weighted_sum(beta) + lam * bits
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot, tb, n = tot + float(loss.detach()), tb + float(bits.detach()), n + 1
        sv = qcode_scores(enc, head, prior, dsets["val"])
        bce = bce_of(sv["s"], dsets["val"].b["succ"])
        if bce < best and ep >= SELECT_FROM:
            best = bce
            best_state = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                          for m in (enc, head, prior)]
        hist.append({"epoch": ep, "loss": tot / n, "train_bits": tb / n, "val_select_bce": bce,
                     "val_bits": float(sv["bits"].mean()), "sec": time.time() - t0})
        log.info("qcode ep %d loss %.3f bits(train) %.1f val bce %.4f val bits %.1f %.0fs", ep,
                 tot / n, tb / n, bce, float(sv["bits"].mean()), time.time() - t0)
    for m, st in zip((enc, head, prior), best_state):
        m.load_state_dict(st)
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "prior": prior.state_dict()}, out / "best.pth")
    scores = {split: qcode_scores(enc, head, prior, dsets[split]) for split in a.splits}
    log.info("selected epoch %d (val BCE %.4f)", SELECT_FROM + int(np.argmin(
        [h["val_select_bce"] for h in hist][SELECT_FROM:])), best)
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
            obs = augment(tr.b["obs"][j].float()).to(DEV)
            R = tr.b["R"][j].to(DEV)
            dt = (tr.b["t"][j] - tr.b["c"][j]).to(DEV)
            Rh, dth = net(obs)
            loss = pose_loss(Rh, dth, R, dt, tr.b["obj_id"][j], syms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(j)
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
        if vloss < best and ep >= SELECT_FROM:
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
def _patches(ds, idx, aug: bool = False):
    obs = ds.b["obs"][idx].float()
    if aug:
        obs = augment(obs)
    depth = obs[:, 3:4].to(DEV)
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
            xp, xs = _patches(tr, j, aug=True)
            y = tr.b["succ"][j].reshape(-1).to(DEV)
            loss = F.binary_cross_entropy_with_logits(net(xp, xs), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        sv = gq_scores(net, dsets["val"])
        vb = bce_of(sv, dsets["val"].b["succ"])
        if vb < best and ep >= SELECT_FROM:
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
    ap.add_argument("--method", choices=["msp", "direct", "pose", "gqcnn", "qcode"])
    ap.add_argument("--rank", type=float, default=0.0, help="C1: within-scene ranking weight")
    ap.add_argument("--distill", type=float, default=0.0, help="C2: teacher target weight")
    ap.add_argument("--lam", type=float, default=0.3, help="C3: weight per entropy-coded bit")
    ap.add_argument("--make-teacher", action="store_true")
    ap.add_argument("--planner", default="", help="C4: plan tag whose scores feed the head")
    ap.add_argument("--vlm-labels", default="", help="study 3: VLM label tag, e.g. "
                    "gemini-robotics-er-2-preview (files labels_<tag>_<split>_v2.pt)")
    ap.add_argument("--vlm-aux", type=float, default=0.0,
                    help="study 3 V1b: auxiliary VLM-score weight, relative to beta")
    ap.add_argument("--vlm-control", default="", choices=["", "const", "shuffle"])
    ap.add_argument("--vlm-mix-ens", action="store_true",
                    help="study 3 V2: distil the mean of the VLM and the C2 ensemble teacher")
    ap.add_argument("--design", action="store_true",
                    help="redesign study: write to results/sr/design and score val/calib only, "
                         "so no test number exists before the design is frozen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=300.0)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--holdout", default="", help="comma-separated obj ids held out of training")
    ap.add_argument("--data", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--points", default="", choices=["", "3d", "4d"],
                    help="study 6: add a point-cloud branch (3d: current frame; 4d: + two "
                         "registered frames with their frame offset as t)")
    a = ap.parse_args()
    if a.make_teacher:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
        make_teacher()
        return
    if a.method is None:
        ap.error("--method is required")
    if a.distill > 0 and a.holdout and not a.vlm_labels:
        ap.error("the C2 teacher saw every object; it cannot be used with --holdout")
    if a.vlm_mix_ens and (a.holdout or not a.distill):
        ap.error("--vlm-mix-ens needs --distill and no --holdout (the ensemble saw every object)")
    if (a.vlm_aux > 0 or a.vlm_control) and not a.vlm_labels:
        ap.error("--vlm-aux/--vlm-control need --vlm-labels")
    if a.points and a.method not in ("msp", "direct"):
        ap.error("--points is implemented for --method msp/direct")
    LMGrasp.points_mode = a.points
    if a.epochs <= SELECT_FROM:
        ap.error(f"--epochs must exceed SELECT_FROM={SELECT_FROM}")
    torch.set_num_threads(a.threads)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    seed_everything(a.seed)
    os.environ.setdefault("TORCH_HOME", "/nas/Selim/selim_sarowar/.cache/torch")

    holdout = {int(x) for x in a.holdout.split(",") if x}
    data = Path(a.data)
    pl = a.planner
    dsets = {"train": load_split("train", data, holdout, unseen=False, distill=a.distill,
                                 planner=pl, a=a),
             "val": load_split("val", data, holdout, unseen=False, planner=pl),
             "calib": load_split("calib", data, holdout, unseen=bool(holdout), planner=pl),
             "test": (load_split("test", data, holdout, unseen=bool(holdout), planner=pl)
                      if not a.design else load_split("val", data, holdout, False, planner=pl))}
    if holdout:  # held-out objects of the val split: the secondary selection check
        dsets["val_unseen"] = load_split("val", data, holdout, unseen=True, planner=pl)
    vl = ("_vlm" + (f"kd{a.distill:g}" if a.distill else "") + (f"aux{a.vlm_aux:g}" if a.vlm_aux else "")
          + ("ens" if a.vlm_mix_ens else "")
          + (f"_{a.vlm_control}" if a.vlm_control else "")) if a.vlm_labels else ""
    variant = (vl + (f"_rank{a.rank:g}" if a.rank else "")
               + (f"_kd{a.distill:g}" if a.distill and not a.vlm_labels else "")
               + (f"_pl{a.planner}" if a.planner else "")
               + (f"_pts{a.points}" if a.points else "")
               + (f"_lam{a.lam:g}" if a.method == "qcode" else ""))
    tag = a.tag or (f"{a.method}" + (f"_b{a.beta:g}_d{a.latent}" if a.method == "msp" else "")
                    + variant
                    + (f"_ho{a.holdout.replace(',', '-')}" if holdout else "") + f"_s{a.seed}")
    a.splits = ("val", "calib") if a.design else ("val", "calib", "test")
    if holdout:
        a.splits += ("val_unseen",)
    out = REPO / ("results/sr/design" if a.design else "results/sr/runs") / tag
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s: train %d val %d calib %d test %d", tag,
             *(len(dsets[k]) for k in ("train", "val", "calib", "test")))
    t0 = time.time()
    if a.method in ("msp", "direct"):
        res = train_msp(a, dsets, out, direct=(a.method == "direct"))
    elif a.method == "pose":
        res = train_pose(a, dsets, out)
    elif a.method == "qcode":
        res = train_qcode(a, dsets, out)
    else:
        res = train_gqcnn(a, dsets, out)
    res["args"] = {k: v for k, v in vars(a).items() if k != "splits"}
    res["wall_seconds"] = time.time() - t0
    res["device"] = str(DEV)
    res["split_index"] = {k: {"obj_id": v.b["obj_id"], "im_id": v.b["im_id"]}
                          for k, v in dsets.items() if k in a.splits}
    torch.save(res, out / "scores.pt")
    (out / "done.json").write_text(json.dumps({"wall_seconds": res["wall_seconds"],
                                               "args": res["args"]}, indent=1))
    log.info("done %s in %.0fs", tag, res["wall_seconds"])


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
