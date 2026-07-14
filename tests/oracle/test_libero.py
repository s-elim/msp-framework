"""The LIBERO oracle: real grocery meshes, and the experiment they make possible.

The load-bearing test is `test_the_analytic_proxy_plans_on_a_different_geometry_than_the_physics`.
The whole point of using scanned objects rather than boxes is that a bounding-box "reconstruction"
and the true shape are DIFFERENT things. On a box they are the same thing, the gap is identically
zero, and the paper's central claim (blueprint wrong-assumption #11) is unfalsifiable.
"""

from __future__ import annotations

import os

import pytest
import torch

mujoco = pytest.importorskip("mujoco")
os.environ.setdefault("MUJOCO_GL", "egl")

from msp.diagnostics import compare_predictors, roc_auc  # noqa: E402
from msp.oracle import AnalyticGraspOracle, LiberoGraspOracle, LiberoObjectLibrary  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    sim = LiberoGraspOracle()
    an = AnalyticGraspOracle(shape="box", noise=False)
    g = torch.Generator().manual_seed(0)
    oi, x = sim.sample_scenes(40, generator=g)
    an.set_base_half(sim.base_half(oi))
    a = an.sample_actions(x, 6, generator=g)
    return sim, an, oi, x, a, sim.query_scenes(oi, x, a)


# --------------------------------------------------------------------------------------
# The assets
# --------------------------------------------------------------------------------------


def test_assets_are_vendored_and_self_contained() -> None:
    """The repository must reproduce a number without a LIBERO checkout. The upstream install is
    only ever READ; nothing here writes to it."""
    lib = LiberoObjectLibrary()
    assert "msp_framework" in str(lib.root), (
        f"loading assets from {lib.root}, not the vendored copy. The experiment would then depend "
        "on a checkout that is not part of this repository."
    )
    assert len(lib.graspable()) >= 10


def test_every_object_compiles_and_is_visible(world) -> None:
    """REGRESSION -- TWO SILENT ASSET FAILURES.

    (a) An object whose visual mesh cannot be resolved renders INVISIBLE while remaining perfectly
        solid: the camera photographs an empty table while the physics grasps something. A corpus of
        pictures of nothing, with labels.
    (b) Several LIBERO assets hard-code an absolute texture path from the original author's machine
        (`/home/yifengz/workspace/...`). That raises only when MuJoCo compiles the model -- i.e.
        deep inside a rollout, mid-corpus.

    So objects are validated by COMPILING them, up front, and the bad ones are dropped.
    """
    sim = LiberoGraspOracle()
    assert sim.n_objects >= 10
    obs = sim.render(
        torch.arange(min(4, sim.n_objects)),
        sim.sample_scenes(min(4, sim.n_objects), generator=torch.Generator().manual_seed(0))[1],
        height=64,
        width=64,
    )
    assert float(obs[:, :3].mean()) > 0.05, "the scene renders black -- nothing is visible"
    assert float(obs[:, :3].std()) > 0.02, "every pixel is identical -- the object is not there"


# --------------------------------------------------------------------------------------
# The physics
# --------------------------------------------------------------------------------------


def test_objects_settle_where_gravity_puts_them(world) -> None:
    """REGRESSION -- THE POSE THE OBJECT IS ACTUALLY IN.

    A parametric box can be placed at a pose you choose. A scanned bottle cannot: dropped on a
    table it topples and comes to rest wherever gravity finds a stable configuration. Sampling a
    nominal yaw and then planning grasps against it means planning for an object that is not there,
    and it had 82% of grasp poses rejected for collision before they were ever simulated -- which
    would have been written up as "real objects are hard to grasp".

    So `sample_scenes` DROPS the object and reads the settled pose back into the state. Some objects
    must therefore end up tipped over: if every one stays upright, the settling is not running.
    """
    sim, _, _, x, _, _ = world
    rot = x[:, 3:6].norm(dim=-1)  # axis-angle magnitude
    assert float(rot.max()) > 0.2, "no object rotated at all -- is the settle step running?"


def test_outcomes_are_valid_and_informative(world) -> None:
    """Assumption A8. If every grasp fails, the corpus carries no information about x."""
    *_, y = world
    y.validate()
    rate = float(y.succ.mean())
    assert 0.15 < rate < 0.9, f"success rate {rate:.3f} is degenerate"
    assert float(y.succ.squeeze(-1).std(dim=1).mean()) > 0.05, (
        "success barely varies across actions; the outcome is not action-conditional"
    )


def test_the_simulator_executes_the_grasp_that_was_PLANNED(world) -> None:
    """REGRESSION -- THE WORST BUG IN THIS REPOSITORY, AND IT LOOKED EXACTLY LIKE PHYSICS.

    `sample_actions` emits a grasp point in the WORLD frame (`t_obj + lift`, where t_obj is the
    object's settled world position). `_rollout` was then adding the object's world position AGAIN::

        hand_pos = rest + action[0:3] - R@TCP   ==   2*t_obj + lift - R@TCP

    so the gripper was displaced by the object's entire world position -- centimetres up, and
    sideways -- and the simulator executed a different grasp from the one that had been planned and
    scored. Nothing crashed. Every symptom read as a physical finding:

      * Tall objects were never gripped, and were left standing on the table. Slip is measured
        against the hand, so an ungrasped object reports slip == lift_height: SIX objects reported a
        mean slip of 0.1185 m against a lift_height of 0.12 m. Identical "physics" across six
        different geometries is not physics.
      * Only the two SHORTEST objects worked at all (ketchup 42%, salad_dressing 26%) -- i.e. the
        two with the smallest displacement.
      * The NOMINAL, zero-noise grasp was the WORST of all (5%), because it is deterministically
        displaced, while sampling noise sometimes cancelled the offset. A proposal whose mode is its
        worst sample is not a proposal; it is a bug.
      * And the analytic tier -- which plans correctly in the world frame -- was scoring one grasp
        while the simulator executed another. That is precisely how a grasp-quality metric measures
        as "uninformative" (AUC 0.539) when it is nothing of the kind. It would have been written up
        as the paper's headline result.

    Nominal success went 6.2% -> 99.6% on the fix. The invariant below is the one that could not
    hold under the bug: aim the jaws exactly where the object is, and the object must come up.
    """
    sim, an, oi, x, _, _ = world
    g = torch.Generator().manual_seed(3)
    a = an.sample_actions(x, 2, generator=g, spread=0.0)  # the NOMINAL grasp: no noise at all
    y = sim.query_scenes(oi, x, a)

    rate = float(y.succ.mean())
    assert rate > 0.9, (
        f"the nominal grasp -- aimed straight at the settled object, with zero proposal noise -- "
        f"succeeds only {rate:.1%} of the time. The gripper is not going where it was told. "
        f"Check the frame convention in LiberoGraspOracle._rollout: the action is a WORLD-frame "
        f"grasp point and must not have the object's position added to it a second time."
    )
    held = y.succ.squeeze(-1) == 1
    assert float(y.slip.squeeze(-1)[held].mean()) < 0.01, "a nominal grasp should not slip"


def test_a_tighter_proposal_produces_better_grasps(world) -> None:
    """REGRESSION. Sanity that the proposal is a proposal: narrowing it toward the ideal antipodal
    grasp must IMPROVE the success rate, monotonically. Under the frame bug this ran backwards --
    spread 1.0 scored 0.094 and spread 0.0 scored 0.050 -- because the noise was partially
    cancelling a deterministic displacement. That inversion is the signature of an aiming error, and
    it is the cheapest possible check for one.
    """
    sim, an, oi, x, _, _ = world
    rates = []
    for spread in (1.0, 0.25, 0.0):
        g = torch.Generator().manual_seed(11)
        a = an.sample_actions(x, 3, generator=g, spread=spread)
        rates.append(float(sim.query_scenes(oi, x, a).succ.mean()))

    assert rates[0] < rates[1] < rates[2], (
        f"success must rise as the proposal tightens; got {rates}. If it FALLS, the proposal's mode "
        f"is worse than its samples, which means the grasp is being aimed at the wrong place."
    )


def test_a_carried_object_does_not_slip(world) -> None:
    *_, y = world
    held = y.succ.squeeze(-1) == 1
    if not bool(held.any()):
        pytest.skip("no successful grasps in this sample")
    assert float(y.slip.squeeze(-1)[held].mean()) < 0.02


# --------------------------------------------------------------------------------------
# THE EXPERIMENT
# --------------------------------------------------------------------------------------


def test_the_analytic_proxy_plans_on_a_different_geometry_than_the_physics(world) -> None:
    """THE REASON REAL OBJECTS ARE NECESSARY.

    The analytic tier plans on an oriented BOUNDING BOX; the simulator collides the object's TRUE
    convex decomposition. Those must actually differ, or there is no hallucinated surface and
    nothing to measure. On a parametric box they would be the same object.
    """
    sim, _, _, _, _, _ = world

    boxlike, complex_ = [], []
    for k in range(sim.n_objects):
        obj = sim.objects[k]
        bbox_vol = float(8.0 * obj.aabb_half_extents().prod())
        decomp_vol = float(8.0 * (obj.box_sizes.prod(axis=1)).sum())
        assert bbox_vol >= decomp_vol - 1e-12, "a bounding box cannot be smaller than what it bounds"
        (boxlike if obj.n_boxes == 1 else complex_).append((obj.name, bbox_vol / decomp_vol))

    # Six of the thirteen groceries have a SINGLE-box collision hull -- butter, cookies, cream
    # cheese and friends simply ARE boxes. For those the reconstruction is EXACT and there is no
    # hallucinated surface at all. That is not a defect; it is a natural control group, and the
    # thesis predicts the analytic proxy should work on them and fail on the curved objects. What
    # would be fatal is if EVERY object were a box, because then there would be nothing to measure
    # -- which is precisely the situation with the parametric box world.
    assert len(complex_) >= 5, (
        f"only {len(complex_)} objects have non-box geometry; the experiment has no signal"
    )
    for name, ratio in complex_:
        assert ratio > 1.05, f"{name}: bounding box over-approximates by only {ratio:.2f}x"


def test_the_proxy_comparison_reports_the_majority_baseline() -> None:
    """Success is rare, so ACCURACY is not informative on its own: always predicting 'fail' scores
    ~90%. The comparison must expose that baseline, or a reader will mistake a useless predictor for
    a good one."""
    torch.manual_seed(0)
    n = 2000
    y = (torch.rand(n) < 0.1).float()
    noise = torch.rand(n)  # a score that knows nothing

    rep = compare_predictors(
        analytic_score=noise, true_success=y, executable=torch.ones(n, dtype=torch.bool)
    )
    assert abs(rep.analytic_auc - 0.5) < 0.05, "a random score must score AUC ~ 0.5"
    assert not rep.analytic_is_informative(), (
        "a random score must not be judged informative -- it cannot beat the majority baseline"
    )
    assert "uninformative" in rep.summary()


def test_auc_is_invariant_to_monotone_rescaling() -> None:
    """AUC is what makes the result immune to the 'you just need a better threshold' rebuttal: it
    does not change under any monotone transform of the score."""
    torch.manual_seed(0)
    s = torch.randn(500).numpy()
    y = (torch.rand(500) < 0.3).float().numpy()
    import numpy as np

    a1 = roc_auc(s, y)
    a2 = roc_auc(3.0 * s + 7.0, y)  # affine
    a3 = roc_auc(np.exp(s), y)  # monotone nonlinear
    assert a1 == pytest.approx(a2, abs=1e-9)
    assert a1 == pytest.approx(a3, abs=1e-9)


def test_non_executable_grasps_are_excluded_from_the_comparison() -> None:
    """Both predictors trivially agree that an unreachable grasp fails. Including those inflates
    every number without saying anything about grasp QUALITY."""
    y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    score = torch.tensor([0.9, 0.1, 0.5, 0.5])
    ex = torch.tensor([True, True, False, False])
    rep = compare_predictors(analytic_score=score, true_success=y, executable=ex)
    assert rep.n_grasps == 2
