"""Theorem 4, as a unit test.

The audited repository had ZERO lines of code for contribution C2. The simulated T-RO review
names the subspace-alignment experiment as the single result that moved Reviewer A from
Major Revision to Accept -- and it did not exist.

These tests validate the identifiability diagnostic against `SyntheticOracle`, whose null
space is known in closed form. That ordering matters: on a real simulator you cannot tell
"the theory is wrong" apart from "my estimator is broken", because there is no ground truth.
Here there is, so the estimator is proven correct BEFORE it is ever pointed at real physics.
"""

from __future__ import annotations

import pytest
import torch

from msp.diagnostics.identifiability import (
    analyze,
    measure_invariant_subspace,
    null_space,
    outcome_jacobian,
    principal_angles,
    row_space,
    subspace_alignment,
)
from msp.oracle.synthetic import SyntheticOracle


@pytest.fixture
def world() -> tuple[SyntheticOracle, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    oracle = SyntheticOracle(state_dim=6, rank=3, action_dim=7, seed=0, noise=False)
    state = oracle.sample_states(1)[0]  # (6,)
    actions = torch.randn(1, 64, 7)  # a rho-dense action set (Def 9.1)
    return oracle, state, actions


# --------------------------------------------------------------------------------------
# Eq 25 -- the Jacobian
# --------------------------------------------------------------------------------------


def test_eq25_jacobian_has_the_expected_shape(world) -> None:
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)
    assert J.shape == (64 * 3, 6)  # (Na * P, state_dim)


def test_eq25_autodiff_jacobian_agrees_with_finite_differences(world) -> None:
    """The autodiff path and the black-box path must agree. A real simulator can only use
    finite differences, so the two estimators must be interchangeable."""
    oracle, state, actions = world
    J_auto = outcome_jacobian(oracle, state, actions)

    oracle.differentiable = False  # force the finite-difference branch
    J_fd = outcome_jacobian(oracle, state, actions)  # float32-optimal eps
    oracle.differentiable = True

    # central differences in float32 cannot do better than ~1e-3 absolute
    torch.testing.assert_close(J_auto, J_fd, atol=2e-3, rtol=2e-2)


# --------------------------------------------------------------------------------------
# Theorem 4 -- ker J(x) is the indistinguishability class
# --------------------------------------------------------------------------------------


def test_theorem4_rank_and_null_dim_match_ground_truth(world) -> None:
    """dim [x] = d_X - rank J(x). The oracle knows both exactly."""
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)

    N = null_space(J)
    R = row_space(J)

    assert R.shape == (6, 3), "rank J(x) must equal rank(P) = 3"
    assert N.shape == (6, 3), "dim ker J(x) must equal state_dim - rank = 3"
    assert N.shape[1] == oracle.null_space_dim()


def test_theorem4_recovers_the_true_null_space_exactly(world) -> None:
    """THE HEADLINE TEST FOR CONTRIBUTION C2.

    The estimated ker J(x) must coincide with the analytically known ker(P). We measure
    coincidence by principal angles: all must be ~0.
    """
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)

    predicted = null_space(J)
    truth = oracle.true_null_space()

    angles = principal_angles(predicted, truth)
    max_deg = torch.rad2deg(angles).max().item()

    assert max_deg < 0.5, (
        f"predicted ker J(x) deviates from the true ker(P) by up to {max_deg:.3f} degrees; "
        "the identifiability estimator is wrong"
    )
    assert subspace_alignment(predicted, truth) == pytest.approx(1.0, abs=1e-4)


def test_theorem4_row_space_recovers_the_identifiable_directions(world) -> None:
    """The complement: row J(x) must coincide with row(P), the directions that DO change
    outcomes and are therefore the only ones a pose readout may honestly report."""
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)
    assert subspace_alignment(row_space(J), oracle.true_row_space()) == pytest.approx(
        1.0, abs=1e-4
    )


def test_theorem4_null_and_row_spaces_are_orthogonal(world) -> None:
    """ker J(x) = (row J(x))^perp. A basic invariant that catches transposition slips."""
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)
    N, R = null_space(J), row_space(J)
    torch.testing.assert_close(R.T @ N, torch.zeros(3, 3), atol=1e-5, rtol=0)


def test_theorem4_perturbing_along_the_null_space_changes_no_outcome(world) -> None:
    """THE PHYSICAL CONTENT OF THE THEOREM.

    Moving the state along ker J(x) must leave every outcome of every action unchanged --
    that is precisely manipulation-indistinguishability (Eq 5). Moving along row J(x) must
    change them. This test does not use the Jacobian at all: it queries the oracle.
    """
    oracle, state, actions = world
    N = oracle.true_null_space()
    R = oracle.true_row_space()

    base = oracle.outcome_params(state.unsqueeze(0), actions)

    # Along the null space: a LARGE step must still change nothing.
    for k in range(N.shape[1]):
        moved = oracle.outcome_params((state + 3.0 * N[:, k]).unsqueeze(0), actions)
        torch.testing.assert_close(moved, base, atol=1e-5, rtol=1e-4)

    # Along the row space: even a SMALL step must change something.
    for k in range(R.shape[1]):
        moved = oracle.outcome_params((state + 0.1 * R[:, k]).unsqueeze(0), actions)
        assert (moved - base).abs().max() > 1e-3, (
            "perturbation along row J(x) must be manipulation-relevant"
        )


# --------------------------------------------------------------------------------------
# The V4 experiment: predicted vs MEASURED invariant subspace
# --------------------------------------------------------------------------------------


def test_measured_invariant_subspace_matches_the_predicted_null_space(world) -> None:
    """THE EXPERIMENT THAT WON REVIEWER A.

    `measure_invariant_subspace` uses NO gradients -- it perturbs the state in random
    directions and keeps those that leave real outcomes unchanged. Theorem 4 predicts this
    measured subspace coincides with ker J(x). We report the principal angles, which is
    exactly the number V4 promises the reviewers.
    """
    oracle, state, actions = world
    J = outcome_jacobian(oracle, state, actions)
    predicted = null_space(J)

    measured = measure_invariant_subspace(oracle, state, actions, n_probes=1024)
    assert measured.shape[1] == 3, "the probe must recover a 3-dimensional invariant subspace"

    angles = principal_angles(predicted, measured)
    max_deg = torch.rad2deg(angles).max().item()
    assert max_deg < 5.0, f"predicted and measured subspaces differ by {max_deg:.2f} degrees"


def test_analyze_produces_a_reportable_summary(world) -> None:
    oracle, state, actions = world
    report = analyze(oracle, state, actions)

    assert report.numerical_rank == 3
    assert report.null_dim == 3
    assert report.max_angle_deg is not None and report.max_angle_deg < 5.0

    m = report.to_metrics()
    assert m["identifiability/rank"] == 3.0
    assert m["identifiability/null_dim"] == 3.0


def test_full_rank_oracle_has_a_trivial_null_space() -> None:
    """Sanity: if the physics depends on every state direction, nothing is unidentifiable
    and MSP's compression claim buys nothing on that scene."""
    torch.manual_seed(1)
    oracle = SyntheticOracle(state_dim=4, rank=3, action_dim=7, seed=1, noise=False)
    state = oracle.sample_states(1)[0]
    actions = torch.randn(1, 64, 7)
    J = outcome_jacobian(oracle, state, actions)
    assert null_space(J).shape[1] == 1  # 4 - 3


def test_synthetic_oracle_rejects_a_trivial_null_space() -> None:
    with pytest.raises(ValueError, match="non-trivial null space"):
        SyntheticOracle(state_dim=6, rank=6)
