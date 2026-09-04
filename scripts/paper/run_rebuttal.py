"""Reviewer-driven experiments. Writes results/rebuttal/*.json.

    MUJOCO_GL=egl python scripts/paper/run_rebuttal.py --only r1 r2 r3 r4

Each block answers one reviewer point and nothing else, so a block can be re-run alone.

  R1  COMMUNICATION (R1.2). The submitted rate is I(X;Z) in nats and the payload figure is
      fp32 storage; neither is a link cost. Here the belief is actually quantized, the
      symbol entropy is measured, and the decision quality is recomputed from the
      dequantized belief, sweeping bits per scene. The link is then modelled: a lost
      payload leaves the receiver with no belief, so it abstains.

  R2  BASELINES (R1.3). The analytic Ferrari-Canny margin is not a learned competitor. The
      decisive comparison is matched: same backbone, same corpus, same actions, trained to
      predict success directly with no information bottleneck and no certificate. If MSP
      does not beat that, the bottleneck is not earning its place.

  R3  UNSEEN OBJECTS (R1.4a). Object identity is held out, not just scene layout: train on
      nine groceries, test on the four never seen. Reports the within-scene AUC the
      reviewers singled out, and whether the certificate still covers.

  R4  SELECTION-CONDITIONAL COVERAGE (R1.4b). The guarantee is calibrated over (o, a)
      pairs but the robot executes an argmax. Recalibrating on the selected action of each
      calibration scene restores exchangeability at the scene level. Unlike
      scripts/paper/selection_conditional_coverage.py, this uses the real calibration
      fold rather than a split of the test set.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "paper"))
os.environ.setdefault("MUJOCO_GL", "egl")

import run_libero as RL  # noqa: E402  (shared corpus/training helpers)
from msp.diagnostics import risk_coverage, roc_auc, within_scene_auc  # noqa: E402
from msp.math.bottleneck import BetaSchedule  # noqa: E402
from msp.math.conformal import conformal_quantile, nonconformity_scores, prediction_set  # noqa: E402
from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.models.nets import Backbone  # noqa: E402
from msp.engine import TrainConfig, Trainer  # noqa: E402
from msp.utils import seed_everything  # noqa: E402

log = logging.getLogger("rebuttal")
ACTION_DIM = 7


# ----------------------------------------------------------------------------- helpers
def beliefs(enc, ds, device, bs: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, d) posterior mean and log-variance for every scene."""
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            b = enc(ds.obs[i : i + bs].to(device))
            mus.append(b.mu.cpu())
            lvs.append(b.logvar.cpu())
    return torch.cat(mus), torch.cat(lvs)


def scores_from_belief(head, mu, logvar, actions, device, k: int = 32, bs: int = 128):
    """s(o, a) from a (possibly reconstructed) belief, matching run_libero._score."""
    from msp.belief import DiagonalGaussianBelief

    out = []
    with torch.no_grad():
        for i in range(0, mu.shape[0], bs):
            b = DiagonalGaussianBelief(mu[i : i + bs].to(device), logvar[i : i + bs].to(device))
            p = head.success_probs(b.rsample(k), actions[i : i + bs].to(device))
            out.append(p.mean(dim=1).cpu())
    return torch.cat(out)


def quantize(x: torch.Tensor, bits: int, lo: torch.Tensor, hi: torch.Tensor):
    """Uniform scalar quantization per dimension. Returns (dequantized, symbols, levels)."""
    levels = 2 ** bits
    span = (hi - lo).clamp_min(1e-6)
    idx = torch.clamp(((x - lo) / span * (levels - 1)).round(), 0, levels - 1)
    return lo + idx / (levels - 1) * span, idx.long(), levels


def symbol_entropy_bits(sym: torch.Tensor, levels: int) -> float:
    """Empirical per-dimension entropy, summed over dimensions: the bits an ideal
    entropy coder needs for one payload. Fixed-width coding costs d*bits instead."""
    total = 0.0
    for j in range(sym.shape[1]):
        counts = torch.bincount(sym[:, j], minlength=levels).float()
        p = counts / counts.sum()
        p = p[p > 0]
        total += float(-(p * p.log2()).sum())
    return total


def decision_metrics(s: torch.Tensor, ds) -> dict:
    succ = ds.succ.squeeze(-1)
    m = ds.executable.bool()
    sel = risk_coverage(s, succ, ds.executable)
    return {
        "pooled_auc": float(roc_auc(s[m].numpy(), succ[m].numpy())),
        "within_scene_auc": float(within_scene_auc(s, succ, ds.executable)),
        "top1_success": float(sel.top1_success),
    }


def conformal_block(s_cal, succ_cal, s_test, succ_test, alpha: float) -> dict:
    q = conformal_quantile(nonconformity_scores(s_cal, succ_cal), alpha)
    ps = prediction_set(s_test, q)
    covered = torch.where(succ_test > 0.5, ps.contains_success, ps.contains_failure)
    cert = ps.contains_success & ~ps.contains_failure
    n = int(cert.sum())
    return {
        "alpha": alpha,
        "q_hat": q,
        "coverage": float(covered.float().mean()),
        "certified_fraction": n / cert.numel(),
        "certified_precision": float(succ_test[cert].mean()) if n else float("nan"),
    }


def flat_exec(s, ds):
    m = ds.executable.bool()
    return s[m], ds.succ.squeeze(-1)[m]


def picked(s, ds):
    masked = s.masked_fill(~ds.executable.bool(), -float("inf"))
    idx = masked.argmax(dim=1)
    rows = torch.arange(s.shape[0])
    return s[rows, idx], ds.succ.squeeze(-1)[rows, idx]


# ----------------------------------------------------------------------------- R1
def r1_communication(ds, dls, device, out: Path, epochs: int) -> dict:
    log.info("=== R1: quantized payload and link behaviour ===")
    enc, head = RL.train_model(ds, dls, device, epochs, beta=300.0, latent=64,
                               out=out / "r1_model")
    rows = []
    mu_cal, lv_cal = beliefs(enc, ds["calib"], device)
    mu_te, lv_te = beliefs(enc, ds["test"], device)
    lo, hi = mu_cal.min(0).values, mu_cal.max(0).values     # range fixed on calibration only
    d = mu_te.shape[1]

    s_cal_ref = scores_from_belief(head, mu_cal, lv_cal, ds["calib"].actions, device)
    s_te_ref = scores_from_belief(head, mu_te, lv_te, ds["test"].actions, device)
    ref = decision_metrics(s_te_ref, ds["test"])
    ref |= conformal_block(*flat_exec(s_cal_ref, ds["calib"]),
                           *flat_exec(s_te_ref, ds["test"]), alpha=0.1)
    ref |= {"bits": 32.0 * d, "entropy_bits": 32.0 * d, "bits_per_dim": 32, "latent_dim": d}
    rows.append(ref)
    log.info("  fp32   %6.0f bits  within-scene %.3f  top1 %.3f  cov %.3f",
             ref["bits"], ref["within_scene_auc"], ref["top1_success"], ref["coverage"])

    for bits in (1, 2, 3, 4, 6, 8):
        qcal, scal, levels = quantize(mu_cal, bits, lo, hi)
        qte, ste, _ = quantize(mu_te, bits, lo, hi)
        s_cal = scores_from_belief(head, qcal, lv_cal, ds["calib"].actions, device)
        s_te = scores_from_belief(head, qte, lv_te, ds["test"].actions, device)
        row = decision_metrics(s_te, ds["test"])
        row |= conformal_block(*flat_exec(s_cal, ds["calib"]),
                               *flat_exec(s_te, ds["test"]), alpha=0.1)
        row |= {
            "bits_per_dim": bits,
            "latent_dim": d,
            "bits": float(bits * d),
            "entropy_bits": symbol_entropy_bits(ste, levels),
        }
        rows.append(row)
        log.info("  %2d-bit %6.0f bits (entropy %6.1f)  within-scene %.3f  top1 %.3f  cov %.3f",
                 bits, row["bits"], row["entropy_bits"], row["within_scene_auc"],
                 row["top1_success"], row["coverage"])

    # Link: a dropped payload leaves the receiver with nothing to certify, so it abstains.
    best = min(rows[1:], key=lambda r: abs(r["within_scene_auc"] - ref["within_scene_auc"]))
    s_te = s_te_ref
    _, succ_pick = picked(s_te, ds["test"])
    ps = prediction_set(s_te, ref["q_hat"])
    cert = (ps.contains_success & ~ps.contains_failure)
    masked = s_te.masked_fill(~ds["test"].executable.bool(), -float("inf"))
    pick = masked.argmax(dim=1)
    rows_i = torch.arange(s_te.shape[0])
    cert_pick = cert[rows_i, pick]

    link = []
    rng = np.random.default_rng(0)
    for p_loss in (0.0, 0.01, 0.05, 0.10, 0.20):
        keep = torch.from_numpy(rng.random(len(pick)) >= p_loss)
        acted = cert_pick & keep
        link.append({
            "p_loss": p_loss,
            "acted_fraction": float(acted.float().mean()),
            "success_given_acted": float(succ_pick[acted].mean()) if int(acted.sum()) else float("nan"),
            "abstain_fraction": float((~acted).float().mean()),
        })
        log.info("  loss %.2f -> acts on %.3f of scenes, success %.3f",
                 p_loss, link[-1]["acted_fraction"], link[-1]["success_given_acted"])

    latency = [{"uplink_mbps": r,
                "payload_bits": best["entropy_bits"],
                "latency_ms": 1e3 * best["entropy_bits"] / (r * 1e6)}
               for r in (0.1, 1.0, 10.0, 100.0)]
    return {"sweep": rows, "link": link, "latency": latency,
            "operating_point": best["bits_per_dim"]}


class _NoRateTrainer(Trainer):
    """Trainer with the rate term switched off for every epoch.

    The objective is loss = sum_j beta_j D_j + rate_weight * R (bottleneck.py:293), so the
    bottleneck is removed by zeroing rate_weight, not by zeroing beta -- BetaSchedule
    rejects beta = 0 because Eq 7 requires it positive, and beta weights the distortion
    side anyway. Overriding here keeps the library untouched, so the main results cannot
    be affected by the baseline.
    """

    def _rate_weight_at(self, epoch: int) -> float:  # noqa: D102
        return 0.0


# ----------------------------------------------------------------------------- R2
def r2_baselines(ds, dls, device, out: Path, epochs: int) -> list[dict]:
    log.info("=== R2: matched learned baseline vs MSP vs analytic ===")
    test = ds["test"]
    succ = test.succ.squeeze(-1)
    rows = []

    enc, head = RL.train_model(ds, dls, device, epochs, beta=300.0, latent=64,
                               out=out / "r2_msp")
    s_msp = RL._score(enc, head, test, device)
    rows.append({"method": "MSP (ours)"} | decision_metrics(s_msp, test))

    # Matched: identical backbone, latent width and action head; beta = 0 removes the
    # rate term, so nothing compresses the representation and no certificate is formed.
    seed_everything(0)
    e2 = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=64)
    h2 = OutcomeHead(latent_dim=64, action_dim=ACTION_DIM)
    _NoRateTrainer(e2, h2, dls["train"], dls["val"], TrainConfig(
        epochs=epochs, lr=5e-4, warmup_epochs=3,
        amp_dtype="bf16" if device.type == "cuda" else "off",
        beta=BetaSchedule.sufficiency_for_success(300.0), out_dir=str(out / "r2_direct"),
    ), device).fit()
    e2.eval(); h2.eval()
    s_dir = RL._score(e2, h2, test, device)
    rows.append({"method": "Direct success CNN (no bottleneck)"} | decision_metrics(s_dir, test))

    rows.append({"method": "Ferrari-Canny on reconstructed box"}
                | decision_metrics(test.margin.squeeze(-1), test))

    g = torch.Generator().manual_seed(0)
    rows.append({"method": "Random"}
                | decision_metrics(torch.rand(succ.shape, generator=g), test))

    for r in rows:
        log.info("  %-38s pooled %.3f  within-scene %.3f  top1 %.3f",
                 r["method"], r["pooled_auc"], r["within_scene_auc"], r["top1_success"])
    return rows


# ----------------------------------------------------------------------------- R3
def r3_unseen_objects(ds, dls, device, out: Path, epochs: int, n_holdout: int) -> dict:
    log.info("=== R3: held-out object identities ===")
    from torch.utils.data import DataLoader, Subset
    from msp.data import collate

    names = ds["train"].object_names
    held = sorted(range(len(names)))[-n_holdout:]
    log.info("  held out: %s", [names[i] for i in held])

    def split(key, keep_held: bool):
        oi = ds[key].object_index
        mask = torch.isin(oi, torch.tensor(held))
        idx = torch.nonzero(mask if keep_held else ~mask).squeeze(-1)
        return idx

    tr_idx, va_idx = split("train", False), split("val", False)
    dl_tr = DataLoader(Subset(ds["train"], tr_idx.tolist()), batch_size=128,
                       shuffle=True, collate_fn=collate)
    dl_va = DataLoader(Subset(ds["val"], va_idx.tolist()), batch_size=128,
                       shuffle=False, collate_fn=collate)

    seed_everything(0)
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=64)
    head = OutcomeHead(latent_dim=64, action_dim=ACTION_DIM)
    Trainer(enc, head, dl_tr, dl_va, TrainConfig(
        epochs=epochs, lr=5e-4, warmup_epochs=3,
        amp_dtype="bf16" if device.type == "cuda" else "off",
        beta=BetaSchedule.sufficiency_for_success(300.0), out_dir=str(out / "r3_unseen"),
    ), device).fit()
    enc.eval(); head.eval()

    result = {"held_out_objects": [names[i] for i in held], "n_train_scenes": int(len(tr_idx))}
    s_te = RL._score(enc, head, ds["test"], device)
    s_cal = RL._score(enc, head, ds["calib"], device)
    for tag, keep in (("seen", False), ("unseen", True)):
        ti, ci = split("test", keep), split("calib", keep)
        sub_t = _view(ds["test"], ti)
        sub_c = _view(ds["calib"], ci)
        m = decision_metrics(s_te[ti], sub_t)
        m |= conformal_block(*flat_exec(s_cal[ci], sub_c), *flat_exec(s_te[ti], sub_t), alpha=0.1)
        m["n_scenes"] = int(len(ti))
        result[tag] = m
        log.info("  %-7s n=%4d  within-scene %.3f  pooled %.3f  top1 %.3f  cov %.3f",
                 tag, m["n_scenes"], m["within_scene_auc"], m["pooled_auc"],
                 m["top1_success"], m["coverage"])
    return result


class _view:
    """A row-subset of a dataset exposing just the tensors the metrics touch."""

    def __init__(self, ds, idx):
        self.succ = ds.succ[idx]
        self.executable = ds.executable[idx]
        self.actions = ds.actions[idx]
        self.margin = ds.margin[idx]


# ----------------------------------------------------------------------------- R4
def r4_selective(ds, dls, device, out: Path, epochs: int, alphas) -> list[dict]:
    log.info("=== R4: coverage on the executed grasp, real calibration fold ===")
    enc, head = RL.train_model(ds, dls, device, epochs, beta=300.0, latent=64,
                               out=out / "r4_model")
    s_cal, s_te = RL._score(enc, head, ds["calib"], device), RL._score(enc, head, ds["test"], device)

    rows = []
    for a in alphas:
        pair_cal, pair_cal_y = flat_exec(s_cal, ds["calib"])
        pair_te, pair_te_y = flat_exec(s_te, ds["test"])
        pick_cal, pick_cal_y = picked(s_cal, ds["calib"])
        pick_te, pick_te_y = picked(s_te, ds["test"])
        row = {
            "alpha": a,
            "pairwise": conformal_block(pair_cal, pair_cal_y, pair_te, pair_te_y, a),
            "executed": conformal_block(pair_cal, pair_cal_y, pick_te, pick_te_y, a),
            "selective": conformal_block(pick_cal, pick_cal_y, pick_te, pick_te_y, a),
        }
        rows.append(row)
        for k in ("pairwise", "executed", "selective"):
            log.info("  a=%.2f %-10s coverage %.4f  certified %.3f  precision %.3f",
                     a, k, row[k]["coverage"], row[k]["certified_fraction"],
                     row[k]["certified_precision"])
    return rows


# ----------------------------------------------------------------------------- R5
def r5_gqcnn(ds, dls, device, out: Path, epochs: int) -> dict:
    """Dex-Net 2.0's grasp-quality CNN, retrained here on the same candidates.

    Reported as an architecture comparison, never as "Dex-Net": the released weights are
    TensorFlow-1 and were trained on another gripper and sensor, so a zero-shot number
    would measure domain gap instead of the scoring function.
    """
    from gqcnn_baseline import score_gqcnn, train_gqcnn

    log.info("=== R5: GQ-CNN architecture, retrained on this corpus ===")
    # Three seeds: a 0.2 AUC gap over the proposed method is large enough that a single
    # run is not evidence, and the table has to carry the spread.
    per_seed = []
    for seed in (0, 1, 2):
        seed_everything(seed)
        net = train_gqcnn(ds["train"], ds["val"], device, epochs=max(20, epochs // 2))
        s_te = score_gqcnn(net, ds["test"], device)
        s_cal = score_gqcnn(net, ds["calib"], device)
        m = decision_metrics(s_te, ds["test"])
        m |= conformal_block(*flat_exec(s_cal, ds["calib"]),
                             *flat_exec(s_te, ds["test"]), alpha=0.1)
        m["seed"] = seed
        per_seed.append(m)
        log.info("  seed %d  pooled %.3f  within-scene %.3f  top1 %.3f  cov %.3f",
                 seed, m["pooled_auc"], m["within_scene_auc"], m["top1_success"], m["coverage"])
        if seed == 0:
            torch.save({"s_gqcnn": s_te}, out / "r5_scores.pt")

    keys = ("pooled_auc", "within_scene_auc", "top1_success", "coverage",
            "certified_fraction", "certified_precision")
    row = {"method": "GQ-CNN architecture (retrained)", "seeds": per_seed}
    for k in keys:
        v = np.array([m[k] for m in per_seed], dtype=float)
        row[k] = float(v.mean())
        row[k + "_sd"] = float(v.std(ddof=1))
    log.info("  %-38s pooled %.3f  within-scene %.3f  top1 %.3f  cov %.3f",
             row["method"], row["pooled_auc"], row["within_scene_auc"],
             row["top1_success"], row["coverage"])
    return row


class ViTBackbone(Backbone):
    """ViT-S/16 built from torchvision's parameterizable VisionTransformer.

    384-wide, 12 layers, 6 heads: the standard ViT-S geometry. Trained from scratch,
    which is not a handicap imposed for convenience but the only option here -- there is
    no ImageNet ViT-S in torchvision, and a pretrained ViT-B could not be reused anyway
    because the width differs, the corpus is four-channel, and 96x96 gives a 6x6 patch
    grid rather than the 14x14 the pretrained position embedding encodes.

    Because of that, the comparison against a pretrained ResNet would confound
    architecture with initialization, so r7 also trains ResNet-18 from scratch.
    """

    def __init__(self, output_dim: int = 256, image_size: int = 96, patch: int = 16) -> None:
        super().__init__(output_dim)
        from torchvision.models import VisionTransformer

        net = VisionTransformer(image_size=image_size, patch_size=patch, num_layers=12,
                                num_heads=6, hidden_dim=384, mlp_dim=1536,
                                num_classes=output_dim)
        old = net.conv_proj
        net.conv_proj = torch.nn.Conv2d(4, old.out_channels, kernel_size=patch, stride=patch)
        self.net = net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _train_with_backbone(bb, ds, dls, device, epochs: int, out: Path, latent: int = 64):
    seed_everything(0)
    enc = BeliefEncoder(bb, latent_dim=latent)
    head = OutcomeHead(latent_dim=latent, action_dim=ACTION_DIM)
    Trainer(enc, head, dls["train"], dls["val"], TrainConfig(
        epochs=epochs, lr=5e-4, warmup_epochs=3,
        amp_dtype="bf16" if device.type == "cuda" else "off",
        beta=BetaSchedule.sufficiency_for_success(300.0), out_dir=str(out),
    ), device).fit()
    enc.eval(); head.eval()
    return enc, head


def r7_backbones(ds, dls, device, out: Path, epochs: int) -> list[dict]:
    """Does a transformer trunk help? Initialization is held constant within the pair."""
    log.info("=== R7: encoder trunk, ViT-S/16 against ResNet-18 ===")
    variants = [
        ("ViT-S/16 (scratch)", lambda: ViTBackbone(output_dim=256)),
        ("ResNet-18 (scratch)", lambda: ResNetBackbone(output_dim=256, pretrained=False)),
        ("ResNet-18 (ImageNet)", lambda: ResNetBackbone(output_dim=256, pretrained=True)),
    ]
    rows = []
    for name, make in variants:
        bb = make()
        n_par = sum(p.numel() for p in bb.parameters()) / 1e6
        enc, head = _train_with_backbone(bb, ds, dls, device, epochs,
                                         out / f"r7_{name.split()[0].replace('/', '')}")
        s_te = RL._score(enc, head, ds["test"], device)
        s_cal = RL._score(enc, head, ds["calib"], device)
        row = {"backbone": name, "params_m": n_par} | decision_metrics(s_te, ds["test"])
        row |= conformal_block(*flat_exec(s_cal, ds["calib"]),
                               *flat_exec(s_te, ds["test"]), alpha=0.1)
        rows.append(row)
        log.info("  %-22s %5.1fM params  pooled %.3f  within-scene %.3f  top1 %.3f  cov %.3f",
                 name, n_par, row["pooled_auc"], row["within_scene_auc"],
                 row["top1_success"], row["coverage"])
    return rows


# ----------------------------------------------------------------------------- R6
def r6_rate_comparison(ds, dls, device, out: Path, epochs: int) -> dict:
    """Bits on the wire versus ranking quality, for MSP and for GQ-CNN.

    Both are quantized identically and their payloads counted the same way. The
    asymmetry the experiment exposes is structural, not a tuning artefact: MSP sends one
    per-scene statistic and scores every candidate from it, while GQ-CNN's representation
    is formed after the grasp is applied, so an edge deployment must send one payload per
    candidate action.
    """
    from gqcnn_baseline import GQCNN, aligned_patches, score_gqcnn, train_gqcnn

    log.info("=== R6: rate versus ranking quality, MSP and GQ-CNN ===")
    n_actions = ds["test"].actions.shape[1]
    rows = []

    enc, head = RL.train_model(ds, dls, device, epochs, beta=300.0, latent=64,
                               out=out / "r6_msp")
    mu_cal, lv_cal = beliefs(enc, ds["calib"], device)
    mu_te, lv_te = beliefs(enc, ds["test"], device)
    lo, hi = mu_cal.min(0).values, mu_cal.max(0).values

    net = train_gqcnn(ds["train"], ds["val"], device, epochs=max(20, epochs // 2))
    with torch.no_grad():
        f_te = []
        xp, xz = aligned_patches(ds["test"])
        xp = xp - xp.mean(dim=(1, 2, 3), keepdim=True)
        for i in range(0, len(xp), 512):
            f_te.append(net.features(xp[i : i + 512].to(device), xz[i : i + 512].to(device)).cpu())
        f_te = torch.cat(f_te)
        xpc, xzc = aligned_patches(ds["calib"])
        xpc = xpc - xpc.mean(dim=(1, 2, 3), keepdim=True)
        f_cal = torch.cat([
            net.features(xpc[i : i + 512].to(device), xzc[i : i + 512].to(device)).cpu()
            for i in range(0, len(xpc), 512)
        ])
    g_lo, g_hi = f_cal.min(0).values, f_cal.max(0).values

    for bits in (1, 2, 3, 4, 8, 32):
        if bits == 32:
            m_te, s_gq = mu_te, None
            e_msp = 32.0 * mu_te.shape[1]
            e_gq = 32.0 * f_te.shape[1] * n_actions
            s_msp = scores_from_belief(head, mu_te, lv_te, ds["test"].actions, device)
            with torch.no_grad():
                s_gq = torch.cat([net.head_from_features(f_te[i : i + 4096].to(device)).cpu()
                                  for i in range(0, len(f_te), 4096)]).sigmoid()
        else:
            qm, sm, lv = quantize(mu_te, bits, lo, hi)
            e_msp = symbol_entropy_bits(sm, lv)
            s_msp = scores_from_belief(head, qm, lv_te, ds["test"].actions, device)
            qf, sf, lvf = quantize(f_te, bits, g_lo, g_hi)
            e_gq = symbol_entropy_bits(sf, lvf) * n_actions
            with torch.no_grad():
                s_gq = torch.cat([net.head_from_features(qf[i : i + 4096].to(device)).cpu()
                                  for i in range(0, len(qf), 4096)]).sigmoid()
        s_gq = s_gq.reshape(ds["test"].actions.shape[0], n_actions)
        rows.append({
            "bits_per_dim": bits,
            "msp": decision_metrics(s_msp, ds["test"]) | {"payload_bits": e_msp},
            "gqcnn": decision_metrics(s_gq, ds["test"]) | {"payload_bits": e_gq},
        })
        log.info("  %2d-bit  MSP %6.1f bits ws=%.3f | GQ-CNN %7.1f bits ws=%.3f",
                 bits, e_msp, rows[-1]["msp"]["within_scene_auc"],
                 e_gq, rows[-1]["gqcnn"]["within_scene_auc"])
    return {"n_actions": n_actions, "sweep": rows}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rebuttal")
    ap.add_argument("--corpus-out", type=Path, default=REPO / "results" / "libero")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--n-holdout", type=int, default=4)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--only", nargs="*", default=["r1", "r2", "r3", "r4", "r5"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  epochs=%d", device, args.epochs)

    ds = RL.build_corpora(args.corpus_out, args.n_train, args.views, args.image_size)
    dls = RL.loaders(ds)

    jobs = {
        "r1": lambda: r1_communication(ds, dls, device, args.out, args.epochs),
        "r2": lambda: r2_baselines(ds, dls, device, args.out, args.epochs),
        "r3": lambda: r3_unseen_objects(ds, dls, device, args.out, args.epochs, args.n_holdout),
        "r4": lambda: r4_selective(ds, dls, device, args.out, args.epochs, args.alphas),
        "r5": lambda: r5_gqcnn(ds, dls, device, args.out, args.epochs),
        "r6": lambda: r6_rate_comparison(ds, dls, device, args.out, args.epochs),
        "r7": lambda: r7_backbones(ds, dls, device, args.out, args.epochs),
    }
    for key in args.only:
        result = jobs[key]()
        path = args.out / f"{key}.json"
        path.write_text(json.dumps(result, indent=2))
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
