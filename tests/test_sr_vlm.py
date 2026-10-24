"""Study 3 (VLM teacher): reply parsing, the v2 agreement score and the teacher controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/sr/vlm"))
sys.path.insert(0, str(REPO / "scripts/sr"))

import gemini_label as gl  # noqa: E402


def test_parse_v1_schema_and_clipping():
    txt = json.dumps({"object": "duck", "material": "rubber", "mass_g": 50,
                      "friction": "high friction", "ratings": [12] + [3] * 15})
    p = gl.parse(txt, 16)
    assert p["ratings"][0] == 10 and p["friction"] == 2 and p["mass_g"] == 50


@pytest.mark.parametrize("txt", ["not json", json.dumps({"ratings": [1, 2]}),
                                 json.dumps({"grasps": []})])
def test_parse_rejects_malformed(txt):
    assert gl.parse(txt, 16) is None


def test_agreement_peaks_at_matching_grasp():
    s = gl.SIZE
    geom = np.array([[s / 2, s / 2, 0.0],    # at the VLM point, same axis
                     [s / 2, s / 2, 90.0],   # same point, orthogonal axis
                     [0.0, 0.0, 0.0]])       # far away
    g = [{"point": [500, 500], "angle_deg": 0}]
    r = gl.agreement(g, geom)
    assert r[0] == pytest.approx(10.0)
    assert r[1] == pytest.approx(5.0)
    assert r[2] < 0.01


def test_agreement_axis_is_undirected():
    s = gl.SIZE
    geom = np.array([[s / 2, s / 2, 10.0]])
    a = gl.agreement([{"point": [500, 500], "angle_deg": 10}], geom)
    b = gl.agreement([{"point": [500, 500], "angle_deg": 190}], geom)
    assert a[0] == pytest.approx(b[0])


def test_vlm_targets_controls(tmp_path, monkeypatch):
    import train_lm

    n, na = 12, 16
    obj = torch.tensor([1] * 6 + [2] * 6)
    succ = (torch.rand(n, na) > 0.5).float()
    data = tmp_path / "data"
    data.mkdir()
    torch.save({"obj_id": obj, "succ": succ}, data / "train.pt")
    r = torch.rand(n, na) * 10
    r[3] = float("nan")  # one unparsed scene
    vdir = tmp_path / "vlm"
    vdir.mkdir()
    torch.save({"ratings": r}, vdir / "labels_m_train_v2.pt")
    monkeypatch.setattr(train_lm, "VLM", vdir)
    base = dict(vlm_labels="m", data=str(data), seed=0)

    t, aux = train_lm.vlm_targets("train", SimpleNamespace(**base, distill=0.5, vlm_aux=0,
                                                           vlm_control=""))
    assert aux is None
    assert torch.equal(t[3], succ[3])  # unparsed scene falls back to the hard label
    assert torch.allclose(t[0], r[0] / 10)

    t, _ = train_lm.vlm_targets("train", SimpleNamespace(**base, distill=0.5, vlm_aux=0,
                                                         vlm_control="const"))
    assert torch.all(t[~torch.isnan(r).any(1)] == 0.5)

    _, aux = train_lm.vlm_targets("train", SimpleNamespace(**base, distill=0, vlm_aux=0.3,
                                                           vlm_control="shuffle"))
    assert not torch.isnan(aux).any()
    for o in (1, 2):  # shuffling stays within an object and keeps the set of scene vectors
        ok = (obj == o) & ~torch.isnan(r).any(1)
        a = sorted(map(tuple, (r[ok] / 10).tolist()))
        b = sorted(map(tuple, aux[ok].tolist()))
        assert np.allclose(a, b)
