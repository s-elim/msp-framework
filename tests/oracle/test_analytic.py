"""The analytic grasp oracle: real wrench-space mechanics, and Theorem 4 on real physics.

The headline test here is `test_theorem4_cylinder_null_space_is_the_symmetry_axis`. It shows the
manipulation-indistinguishability class of Theorem 4 is a MEASURED property of grasp physics, not
an artefact of a synthetic construction: rotating a cylinder about its own axis changes no grasp
outcome, and the outcome Jacobian knows it.

`test_theorem4_is_vacuous_for_a_generic_box` is the honest companion. For a box, J(x) has full
rank and nothing is continuously unidentifiable, so Theorem 4 says nothing at all. The theorem has
content only for objects with a genuine outcome-invariance. That limitation belongs in the paper.
"""

from __future__ import annotations

import math

import pytest
import torch

from msp.diagnostics import measure_invariant_subspace, null_space, outcome_jacobian, principal_angles
from msp.oracle import STATE_DIM, AnalyticGraspOracle


@pytest.fixture
def world():
    torch.manual_seed(0)
    o = AnalyticGraspOracle(shape="box", noise=False)
    g = torch.Generator().manual_seed(0)
    x = o.sample_states(96, generator=g)
    a = o.sample_actions(x, 32, generator=g)
    return o, x, a


# --------------------------------------------------------------------------------------
# The physics is physics
# --------------------------------------------------------------------------------------


def test_outcomes_are_valid_members_of_Y(world) -> None:
    """Y = {0,1} x R x R_{>=0}. `query` calls validate(); an oracle emitting negative slip or a
    non-binary success is a bug that must never reach the loss."""
    o, x, a = world
    y = o.query(x, a)
    y.validate()
    assert y.succ.shape == (96, 32, 1)


def test_a8_outcomes_are_informative(world) -> None:
    """Assumption A8: the success functional must be NONCONSTANT in the action. If every grasp
    succeeds (or every grasp fails) the task is trivial and the dataset carries no information
    about x. This is the assumption that a bad action sampler silently breaks."""
    o, x, a = world
    y = o.query(x, a)
    rate = float(y.succ.mean())
    assert 0.02 < rate < 0.98, f"success rate {rate:.3f} -- the task is degenerate"

    per_scene_std = y.succ.squeeze(-1).std(dim=1).mean()
    assert per_scene_std > 0.05, (
        f"success barely varies across actions (std={per_scene_std:.3f}); the outcome is not "
        "action-conditional and there is nothing for the head to learn"
    )


def test_point_contacts_cannot_resist_torsion_about_the_grasp_axis() -> None:
    """REAL GRASP MECHANICS, stated precisely.

    Take an IDEAL antipodal grasp: two point contacts at r1 = -r2, collinear with the centre of
    mass. Every generator's torque is r x f, which is perpendicular to r, so the torsional
    component ABOUT THE GRASP AXIS is identically zero for every contact force. The wrench hull
    therefore lies inside a coordinate hyperplane, the origin sits on its boundary, and the grasp
    is NOT force-closed: eps = 0 exactly. Physically, the object is free to unscrew itself in the
    hand.

    Soft-finger contact (a real fingertip makes a PATCH, not a point) resists a torsional moment
    up to gamma * mu * f_n about the normal, which lifts the hull out of that hyperplane and makes
    the grasp closable. This is why parallel-jaw grippers work.

    NOTE ON SCOPE -- an earlier version of this test asserted the degeneracy held across the whole
    population, and it does not. The collinearity r1 = -r2 is what kills the torsion, and a random
    COM offset breaks it: off-axis contacts DO generate torsion about the grasp axis even with
    point contacts. The theorem is about the idealized centred case, so that is what we test.
    """
    o_point = AnalyticGraspOracle(shape="box", torsional_coeff=0.0, noise=False)
    o_soft = AnalyticGraspOracle(shape="box", torsional_coeff=0.02, noise=False)

    # A perfectly centred cube, COM at the centroid, grasped exactly along +x.
    x = torch.zeros(1, STATE_DIM)
    x[:, 6:9] = math.log(0.03)  # log_size
    x[:, 9] = math.log(0.6)  # log_friction
    x[:, 10] = math.log(0.3)  # log_mass
    # com stays exactly 0 -> the contacts are collinear with it
    a = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])

    pts, nrm, hit, _ = o_point._contacts(x, a)
    assert bool(hit[0, 0])
    n1, n2 = nrm[0, 0, 0], nrm[0, 0, 1]
    assert float(n1 @ n2) == pytest.approx(-1.0, abs=1e-3), (
        "the two jaw normals must be ANTIPODAL. If they point the same way the grasp geometry is "
        "wrong and every wrench-space number downstream is meaningless."
    )

    eps_point, _ = o_point._epsilon_quality(x, a)
    eps_soft, _ = o_soft._epsilon_quality(x, a)

    assert float(eps_point) == pytest.approx(0.0, abs=1e-4), (
        f"an ideal point-contact grasp must score exactly eps = 0 (not force-closed); "
        f"got {float(eps_point):+.5f}. If this is positive, the direction set has missed the "
        f"degenerate torsional axis -- see _wrench_directions."
    )
    assert float(eps_soft) > 0.05, "soft-finger torsion must make the ideal grasp force-closed"


def test_soft_finger_contact_improves_force_closure_on_the_population() -> None:
    """The population version of the claim above: torsional friction can only help."""
    g = torch.Generator().manual_seed(0)
    soft = AnalyticGraspOracle(shape="box", torsional_coeff=0.02, noise=False)
    x = soft.sample_states(128, generator=g)
    a = soft.sample_actions(x, 32, generator=g)

    point = AnalyticGraspOracle(shape="box", torsional_coeff=0.0, noise=False)
    eps_soft, _ = soft._epsilon_quality(x, a)
    eps_point, _ = point._epsilon_quality(x, a)

    assert torch.all(eps_soft >= eps_point - 1e-5), "torsional friction cannot reduce quality"
    assert float((eps_soft > 0).float().mean()) > float((eps_point > 0).float().mean())


def test_epsilon_is_higher_for_a_centred_grasp_than_an_edge_grasp(world) -> None:
    """Sanity that the quality metric tracks grasp quality: a grasp through the object's centre
    should beat one that barely catches an edge."""
    o, x, _ = world
    x = x[:1]
    t = x[:, 0:3]

    centred = torch.cat([t, torch.tensor([[1.0, 0.0, 0.0, 0.0]])], dim=-1).unsqueeze(1)
    offset = torch.cat(
        [t + torch.tensor([[0.0, 0.028, 0.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])], dim=-1
    ).unsqueeze(1)

    eps_c, _ = o._epsilon_quality(x, centred)
    eps_o, _ = o._epsilon_quality(x, offset)
    assert float(eps_c) > float(eps_o)


# --------------------------------------------------------------------------------------
# Theorem 4 on real grasp physics
# --------------------------------------------------------------------------------------


def test_jacobian_is_differentiable_through_the_wrench_metric(world) -> None:
    """J(x) must come from autodiff. Section 11: 'estimate J(x) by autodiff of Phi in sim'."""
    o, x, a = world
    J = outcome_jacobian(o, x[0], a[:1])
    assert J.shape == (32 * 3, STATE_DIM)
    assert torch.all(torch.isfinite(J))
    assert J.abs().max() > 0, "the Jacobian is identically zero -- Phi does not depend on x"


def test_theorem4_is_vacuous_for_a_generic_box(world) -> None:
    """THE HONEST FINDING. For a box, every state direction -- pose, size, friction, mass, COM --
    changes some outcome of some grasp. J(x) has full rank, ker J(x) = {0}, and Theorem 4 has NO
    CONTENT: nothing is continuously unidentifiable.

    This limits the theorem's reach and belongs in the paper, not in a footnote. The ambiguity a
    box actually has is DISCRETE (its symmetry group), which is Theorem 5, not Theorem 4.
    """
    o, x, a = world
    J = outcome_jacobian(o, x[0], a[:1])
    assert null_space(J).shape[1] == 0, (
        "a generic box should have a trivial null space; if this fails, the box has acquired a "
        "continuous outcome-invariance and the claim above needs rewriting"
    )


def test_theorem4_cylinder_null_space_is_the_symmetry_axis() -> None:
    """THE HEADLINE TEST FOR C2 ON REAL PHYSICS.

    A cylinder is invariant under rotation about its own axis. Every grasp outcome is therefore
    invariant too, so that rotation is EXACTLY a manipulation-indistinguishable direction (Eq 5).
    Theorem 4 predicts it appears as a one-dimensional ker J(x). It does.
    """
    torch.manual_seed(0)
    o = AnalyticGraspOracle(shape="cylinder", noise=False)
    g = torch.Generator().manual_seed(0)
    x = o.sample_states(8, generator=g)
    a = o.sample_actions(x, 48, generator=g)

    J = outcome_jacobian(o, x[0], a[:1])
    N = null_space(J)
    assert N.shape[1] == 1, f"expected a 1-D null space for a cylinder; got {N.shape[1]}"

    # THE PHYSICAL CONTENT: move the state a LONG way along that direction and the outcome of
    # every one of the 48 candidate grasps must be unchanged.
    base = o.outcome_params(x[:1], a[:1])
    for step in (0.5, 1.0, 2.0):
        moved = o.outcome_params((x[0] + step * N[:, 0]).unsqueeze(0), a[:1])
        torch.testing.assert_close(moved, base, atol=2e-3, rtol=1e-2)


def test_measured_invariant_subspace_agrees_on_real_physics() -> None:
    """The gradient-free probe must recover the same subspace as the Jacobian, on real grasp
    mechanics rather than on the synthetic world. This is the V4 perturbation-invariance experiment."""
    torch.manual_seed(0)
    o = AnalyticGraspOracle(shape="cylinder", noise=False)
    g = torch.Generator().manual_seed(0)
    x = o.sample_states(4, generator=g)
    a = o.sample_actions(x, 48, generator=g)

    predicted = null_space(outcome_jacobian(o, x[0], a[:1]))
    measured = measure_invariant_subspace(o, x[0], a[:1], n_probes=3000)

    assert measured.shape[1] >= 1, "the probe found no invariant direction"
    ang = torch.rad2deg(principal_angles(predicted, measured)).max()
    assert float(ang) < 5.0, f"predicted and measured subspaces differ by {float(ang):.1f} deg"


# --------------------------------------------------------------------------------------
# The observation is honestly incomplete
# --------------------------------------------------------------------------------------


def test_observation_does_not_reveal_friction_mass_or_com(world) -> None:
    """The physically honest part. Friction, mass and centre of mass decide slip and torque
    failures, and NO CAMERA OBSERVES THEM. The belief must therefore carry genuine epistemic
    uncertainty about them, which is exactly what active touch (Eq 20/21) exists to resolve and
    what a pose estimator cannot even express.

    Two states differing only in friction/mass/COM must produce the same observation.
    """
    o, x, _ = world
    x2 = x.clone()
    x2[:, 9:] += 1.0  # perturb log_friction, log_mass, com

    torch.testing.assert_close(o.observe(x, noise=0.0), o.observe(x2, noise=0.0))

    # ...but they must produce DIFFERENT outcomes, or the variables would not matter.
    a = o.sample_actions(x, 16, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(o.outcome_params(x, a), o.outcome_params(x2, a))


def test_rejects_an_unknown_shape() -> None:
    with pytest.raises(ValueError, match="box.*cylinder"):
        AnalyticGraspOracle(shape="teapot")
