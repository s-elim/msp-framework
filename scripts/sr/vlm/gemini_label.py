"""Gemini grasp-outcome ratings for LM-Grasp scenes (study 3: a VLM as an outcome teacher).

One call per scene: the real LINEMOD RGB crop with the 16 candidate grasps drawn and numbered,
plus a text table of observation-derived geometry (approach tilt, TCP height above the fitted
table). The VLM returns a 0-10 rating per candidate and object-level guesses (name, material,
mass, friction class). Nothing the learned scorers do not see enters the prompt.

    python scripts/sr/vlm/gemini_label.py --split val --pilot 4 --model gemini-3.6-flash
    python scripts/sr/vlm/gemini_label.py --split train --model <chosen> --workers 8

Reproducibility: temperature 0, pinned model string, prompt in prompt_v1.txt, every response
cached on disk under results/sr/vlm/cache/<sha256>.json keyed by (model, generation config,
prompt text, image bytes). A rerun reads the cache and makes no call. Spend: every billed call
appends to results/sr/vlm/spend.jsonl; the run stops before a call that would take the
cumulative total over --cap. The response is parsed as data against a fixed schema; it is never
executed or followed as an instruction. The API key comes from the shared secrets store and is
sent only in a request header; it never enters a log line, a cache file or an exception.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from secrets_loader import gemini_api_key  # noqa: E402

OUT = REPO / "results/sr/vlm"
CACHE = OUT / "cache"
LEDGER = OUT / "spend.jsonl"
PROMPTS = {"v1": Path(__file__).resolve().parent / "prompt_v1.txt",  # rate drawn candidates
           "v2": Path(__file__).resolve().parent / "prompt_v2_points.txt"}  # point to grasps
SIGMA = 0.08  # v2 agreement kernel width, fraction of the image side (fixed before any v2 call)
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
# USD per 1M tokens, paid tier, https://ai.google.dev/gemini-api/docs/pricing fetched 2026-10-24
# (prices valid to 2026-12-31). Thinking tokens are billed at the output rate.
PRICES = {"gemini-robotics-er-2-preview": (1.00, 5.00), "gemini-3.6-flash": (0.75, 3.75),
          "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50)}
FRICTION = {"low": 0, "medium": 1, "high": 2}
HALF_AXIS = 0.025  # drawn half-length of the closing-axis line, metres
SIZE = 448  # side of the image sent to the VLM, pixels
LM_ROOT = REPO / "data/lm"
log = logging.getLogger("vlm")
_lock = threading.Lock()


def _font(px: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if Path(f).exists():
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def render(b: dict, i: int, draw: bool = True) -> tuple[bytes, str, np.ndarray]:
    """PNG of the real RGB crop (with numbered candidates if `draw`), the per-candidate text
    table, and each candidate's image geometry (u, v, closing-axis angle in degrees, 0-180)."""
    o, im = int(b["obj_id"][i]), int(b["im_id"][i])
    rgb = Image.open(LM_ROOT / f"test/{o:06d}/rgb/{im:06d}.png").convert("RGB")
    u0, v0, side = (float(x) for x in b["crop"][i])
    pad = int(side)
    canvas = Image.new("RGB", (rgb.width + 2 * pad, rgb.height + 2 * pad))
    canvas.paste(rgb, (pad, pad))
    crop = canvas.crop((int(u0) + pad, int(v0) + pad, int(u0) + pad + int(side),
                        int(v0) + pad + int(side))).resize((SIZE, SIZE), Image.BICUBIC)
    K = b["K"][i].numpy().astype(np.float64)
    s = SIZE / side

    def px(p: np.ndarray) -> tuple[float, float]:
        return ((K[0, 0] * p[0] / p[2] + K[0, 2] - u0) * s,
                (K[1, 1] * p[1] / p[2] + K[1, 2] - v0) * s)

    d = ImageDraw.Draw(crop)
    font = _font(15)
    cmap = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
            (213, 94, 0), (204, 121, 167), (255, 255, 255)]
    up = b["n_table"][i].numpy().astype(np.float64)
    c = b["c"][i].numpy().astype(np.float64)
    h_tab = float(b["h_table"][i])
    # table-plane fit failed (about a third of scenes): up is the camera -y axis and heights are
    # measured from the object's visible centroid; the table says which reference applies
    ref = "object centre" if h_tab == 0.0 and up[1] == -1.0 else "table"
    rows = []
    T = b["T"][i].numpy().astype(np.float64)
    placed: list[tuple[float, float]] = []
    geom = np.zeros((T.shape[0], 3))
    for k in range(T.shape[0]):
        p, xax, zax = T[k, :3, 3], T[k, :3, 0], T[k, :3, 2]
        a, bb = px(p - HALF_AXIS * xax), px(p + HALF_AXIS * xax)
        cx, cy = px(p)
        # image angle counter-clockwise from horizontal (image v points down), undirected
        geom[k] = cx, cy, np.degrees(np.arctan2(a[1] - bb[1], bb[0] - a[0])) % 180.0
        tilt = np.degrees(np.arccos(np.clip(float(zax @ up), -1.0, 1.0)))
        height = 1000.0 * (h_tab + float((p - c) @ up))
        rows.append(f"{k + 1}: tilt {tilt:.0f} deg, height {height:.0f} mm above {ref}")
        if not draw:
            continue
        col = cmap[k % len(cmap)]
        d.line([a, bb], fill=col, width=3)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=col, outline=(0, 0, 0))
        # greedy label placement: the offset farthest from labels already drawn
        offs = [(6, -18), (6, 2), (-22, -18), (-22, 2), (-8, -26), (-8, 8), (14, -8), (-30, -8)]
        lx, ly = max(((cx + ox, cy + oy) for ox, oy in offs),
                     key=lambda q: min([np.hypot(q[0] - x, q[1] - y) for x, y in placed] or [1e9]))
        placed.append((lx, ly))
        d.line([(cx, cy), (lx + 5, ly + 8)], fill=(255, 255, 255), width=1)
        d.text((lx, ly), str(k + 1), fill=(255, 255, 255), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "\n".join(rows), geom


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line)


def gen_config(model: str) -> dict:
    cfg = {"temperature": 0.0, "responseMimeType": "application/json", "maxOutputTokens": 4096}
    cfg["thinkingConfig"] = {"thinkingLevel": "low"}
    return cfg


def agreement(grasps: list, geom: np.ndarray) -> np.ndarray:
    """v2 teacher score in [0, 10] per candidate: max over the VLM's grasps of a Gaussian
    kernel on image distance times (0.5 + 0.5 |cos| of the closing-axis angle difference)."""
    best = np.zeros(len(geom))
    for g in grasps[:5]:
        y, x = (float(v) * SIZE / 1000.0 for v in g["point"])
        ang = float(g["angle_deg"])
        d2 = (geom[:, 0] - x) ** 2 + (geom[:, 1] - y) ** 2
        k = np.exp(-d2 / (2 * (SIGMA * SIZE) ** 2))
        k *= 0.5 + 0.5 * np.abs(np.cos(np.radians(geom[:, 2] - ang)))
        best = np.maximum(best, k)
    return 10.0 * best


def parse(text: str, n: int, geom: np.ndarray | None = None) -> dict | None:
    """Validate the reply against the fixed schema; anything else is a parse failure."""
    try:
        j = json.loads(text)
        if geom is not None:  # v2
            r = agreement(j["grasps"], geom)
            if not len(j["grasps"]):
                return None
        else:
            r = [float(x) for x in j["ratings"]]
        if len(r) != n:
            return None
        fr = str(j.get("friction", "")).split()[0].lower() if j.get("friction") else ""
        return {"ratings": np.clip(r, 0, 10).tolist(), "object": str(j.get("object", ""))[:64],
                "material": str(j.get("material", ""))[:64],
                "mass_g": float(j["mass_g"]) if j.get("mass_g") is not None else float("nan"),
                "friction": FRICTION.get(fr, -1)}
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def call(model: str, png: bytes, table: str, cap: float, key: str, pv: str = "v1") -> dict:
    import requests

    prompt = PROMPTS[pv].read_text()
    prompt = prompt.format(table=table) if pv == "v1" else prompt
    cfg = gen_config(model)
    h = hashlib.sha256(json.dumps([model, cfg, prompt]).encode() + png).hexdigest()
    f = CACHE / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    body = {"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"text": prompt}]}], "generationConfig": cfg}
    pin, pout = PRICES[model]
    for attempt in range(8):
        with _lock:
            if spent() + 0.05 > cap:
                raise SystemExit(f"spend cap reached: {spent():.2f} of {cap:.2f} USD")
        t0 = time.time()
        try:
            r = requests.post(ENDPOINT.format(m=model), json=body, timeout=180,
                              headers={"x-goog-api-key": key})
        except requests.RequestException as e:
            log.warning("request error %s, retry %d", type(e).__name__, attempt)
            time.sleep(min(60, 2 ** attempt))
            continue
        lat = time.time() - t0
        if r.status_code in (429, 500, 502, 503, 504):
            log.warning("HTTP %d, backoff %d", r.status_code, attempt)
            time.sleep(min(120, 5 * 2 ** attempt))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300].replace(key, '[R]')}")
        j = r.json()
        u = j.get("usageMetadata", {})
        tin = int(u.get("promptTokenCount", 0))
        tout = int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
        usd = (tin * pin + tout * pout) / 1e6
        with _lock:
            with LEDGER.open("a") as fh:
                fh.write(json.dumps({"t": time.time(), "model": model, "key": h, "usd": usd,
                                     "in": tin, "out": tout}) + "\n")
        parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        rec = {"model": model, "model_version": j.get("modelVersion"), "config": cfg,
               "text": text, "usage": u, "usd": usd, "latency_s": lat,
               "finish": (j.get("candidates") or [{}])[0].get("finishReason")}
        f.write_text(json.dumps(rec))
        return rec
    raise RuntimeError("retries exhausted")


def select(b: dict, pilot: int, seed: int = 0) -> np.ndarray:
    if not pilot:
        return np.arange(len(b["obj_id"]))
    rng = np.random.default_rng(seed)
    oid = b["obj_id"].numpy()
    return np.sort(np.concatenate([rng.choice(np.nonzero(oid == o)[0], pilot, replace=False)
                                   for o in np.unique(oid)]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="gemini-3.6-flash", choices=sorted(PRICES))
    ap.add_argument("--pilot", type=int, default=0, help="N scenes per object (0: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--cap", type=float, default=100.0, help="cumulative USD cap, all runs")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    b = torch.load(REPO / f"data/lm_grasp/{a.split}.pt", map_location="cpu", weights_only=False)
    idx = select(b, a.pilot)
    key = gemini_api_key()
    na = b["T"].shape[1]
    log.info("%s %s: %d scenes, spent so far %.4f USD", a.model, a.split, len(idx), spent())

    geoms = {}

    def one(i: int) -> tuple[int, dict]:
        png, table, geoms[int(i)] = render(b, int(i), draw=a.prompt == "v1")
        return int(i), call(a.model, png, table, a.cap, key, a.prompt)

    recs = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (i, rec) in enumerate(ex.map(one, idx)):
            recs[i] = rec
            if n % 200 == 0:
                log.info("%d/%d, spent %.4f USD", n, len(idx), spent())
    R = np.full((len(b["obj_id"]), na), np.nan, np.float32)
    mass = np.full(len(b["obj_id"]), np.nan, np.float32)
    fric = np.full(len(b["obj_id"]), -1, np.int64)
    names, usd, lat, tin, tout, ok = {}, [], [], [], [], 0
    for i, rec in recs.items():
        p = parse(rec["text"], na, geoms[i] if a.prompt == "v2" else None)
        usd.append(rec["usd"])
        lat.append(rec["latency_s"])
        tin.append(int(rec["usage"].get("promptTokenCount", 0)))
        tout.append(int(rec["usage"].get("candidatesTokenCount", 0))
                    + int(rec["usage"].get("thoughtsTokenCount", 0)))
        if p is None:
            continue
        ok += 1
        R[i], mass[i], fric[i] = p["ratings"], p["mass_g"], p["friction"]
        names[i] = (p["object"], p["material"])
    tag = f"{a.model}_{a.split}" + ("" if a.prompt == "v1" else f"_{a.prompt}") + (f"_pilot{a.pilot}" if a.pilot else "")
    torch.save({"ratings": torch.from_numpy(R), "mass_g": torch.from_numpy(mass),
                "friction": torch.from_numpy(fric), "names": names, "idx": idx,
                "model": a.model, "prompt": PROMPTS[a.prompt].name,
                "model_version": sorted({r.get("model_version") for r in recs.values()} - {None})},
               OUT / f"labels_{tag}.pt")
    summ = {"model": a.model, "split": a.split, "n": len(idx), "parsed": ok,
            "usd_total": float(np.sum(usd)), "usd_per_call": float(np.mean(usd)),
            "tokens_in_mean": float(np.mean(tin)), "tokens_out_mean": float(np.mean(tout)),
            "latency_median_s": float(np.median(lat)), "cumulative_usd": spent()}
    (OUT / f"summary_{tag}.json").write_text(json.dumps(summ, indent=1))
    log.info("%s", summ)


if __name__ == "__main__":
    main()
