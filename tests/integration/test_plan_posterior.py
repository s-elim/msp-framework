"""The posterior planner reduces to the original point-estimate planner (study 4 regression).

Needs the BOP LM data in data/lm and the CoACD cache in data/lm_grasp/coacd; skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
LM, CACHE = REPO / "data/lm", REPO / "data/lm_grasp/coacd"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (LM / "test").exists() or not CACHE.exists(),
                                 reason="LM data or CoACD cache missing")]


def test_one_identity_sample_equals_the_point_estimate_planner():
    from msp.data.linemod import crop_observation, index_lm, object_frame, propose_grasps
    from msp.grasp import score_under_posterior
    from msp.oracle.linemod_sim import LinemodGraspSim, decompose_lm_models

    obj = decompose_lm_models(LM, CACHE, obj_ids=(1,))[0]
    sim = LinemodGraspSim(obj)
    e = index_lm(LM, (1,))[0]
    rng = np.random.default_rng([0, e.obj_id, e.im_id])
    ob = crop_observation(LM, e)
    prop = propose_grasps(ob["points"], ob["c"], 8, rng, n_table=ob["n_table"],
                          h_table=float(ob["h_table"]))
    prng = np.random.default_rng(7)
    phys = [sim.nominal_phys(), sim.sample_phys(prng)]
    # original planner (scripts/sr/plan_with_pose.py::_score), written out
    T_obj = object_frame(prop.T, e.R, e.t)
    ref = np.array([[sim.rollout(T, p)[0] for T in T_obj] for p in phys]).mean(0)
    post = score_under_posterior(sim.rollout, prop.T, e.R[None], e.t[None], phys)
    assert np.array_equal(post.mean, ref)
    # labels were generated at the annotated pose: the planner at that pose sees the same
    # executability as the corpus builder for the nominal physics draw
    exe = np.array([sim.rollout(T, phys[0])[2] for T in T_obj])
    assert np.array_equal(post.exe[0, 0], exe)


def test_rollout_trace_leaves_the_outcome_unchanged():
    from msp.data.linemod import crop_observation, index_lm, object_frame, propose_grasps
    from msp.oracle.linemod_sim import LinemodGraspSim, decompose_lm_models

    sim = LinemodGraspSim(decompose_lm_models(LM, CACHE, obj_ids=(1,))[0])
    e = index_lm(LM, (1,))[0]
    ob = crop_observation(LM, e)
    prop = propose_grasps(ob["points"], ob["c"], 8, np.random.default_rng([0, 1, e.im_id]),
                          n_table=ob["n_table"], h_table=float(ob["h_table"]))
    p = sim.nominal_phys()
    rest = sim.rest_pose(p)[0]
    for T in object_frame(prop.T, e.R, e.t):
        trace: list[np.ndarray] = []
        succ, slip, exe = sim.rollout(T, p, trace=trace)
        assert (succ, slip, exe) == sim.rollout(T, p)
        assert len(trace) == (3 if exe else 1)
        assert all(q.shape == (sim.model.nq,) for q in trace)
        if exe:  # the last state is the one the success test reads
            lifted = trace[-1][sim._oa + 2] > rest[2] + sim.s.success_height
            assert bool(lifted) == bool(succ)
        dense: list[np.ndarray] = []
        assert sim.rollout(T, p, trace=dense, trace_every=50) == (succ, slip, exe)
        n_steps = 20 + sim.s.close_steps + sim.s.hold_steps + sim.s.lift_steps
        assert len(dense) == (1 + n_steps // 50 if exe else 1)
        if exe and n_steps % 50 == 0:
            assert np.array_equal(dense[-1], trace[-1])
