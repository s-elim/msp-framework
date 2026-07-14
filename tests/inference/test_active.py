"""Active perception: Eq 16, 17, 18, and belief fusion. Contribution C3, first half.

The load-bearing test is `test_fusing_a_second_view_reduces_ambiguity`. If a second look does not
reduce the epistemic variance, then IG is zero for every viewpoint, the acquisition network has
nothing to learn, and active perception is decoration. Everything else in this file is downstream
of that one fact.
"""

from __future__ import annotations

import pytest
import torch

from msp.belief import DiagonalGaussianBelief
from msp.inference import ActiveConfig, ActivePerception
from msp.math.voi import acquisition_loss, information_gain
from msp.models import AcquisitionNet, BeliefEncoder, MLPBackbone


# --------------------------------------------------------------------------------------
# Belief fusion -- the thing Eq 17's `o ∪ o_b` actually means
# --------------------------------------------------------------------------------------


def test_fusing_independent_views_sharpens_the_belief() -> None:
    """THE POINT OF FUSION. Two independent views of the same object must make you MORE certain.

    A fusion that merely averages the means leaves the variance untouched -- and since epistemic
    variance is the ENTIRE signal driving Eq 16 and Eq 17, a fusion that cannot reduce it makes the
    information gain of every viewpoint identically zero, and active perception a no-op.
    """
    b1 = DiagonalGaussianBelief(torch.zeros(2, 4), torch.zeros(2, 4))  # sigma^2 = 1
    b2 = DiagonalGaussianBelief(torch.zeros(2, 4), torch.zeros(2, 4))

    fused = DiagonalGaussianBelief.fuse([b1, b2])

    # precisions: 1 + 1 - (k-1)*1 = 1... the prior is subtracted once. Two views each carrying
    # unit precision, on a unit prior, leave unit precision -- they were uninformative.
    torch.testing.assert_close(fused.logvar, torch.zeros(2, 4))

    # Now two genuinely INFORMATIVE views (precision 4 each):
    sharp = DiagonalGaussianBelief(torch.zeros(2, 4), torch.full((2, 4), -1.386))  # var 0.25
    fused2 = DiagonalGaussianBelief.fuse([sharp, sharp])
    assert torch.all(fused2.logvar < sharp.logvar), "two informative views must sharpen the belief"

    # precision: 4 + 4 - 1 = 7  =>  var = 1/7
    torch.testing.assert_close(
        torch.exp(fused2.logvar), torch.full((2, 4), 1.0 / 7.0), atol=2e-3, rtol=1e-2
    )


def test_fusion_subtracts_the_prior_exactly_once() -> None:
    """Without the (k-1) prior subtraction, k views of a COMPLETELY UNINFORMATIVE scene would still
    collapse the belief to a point -- you would become certain by taking photographs of nothing."""
    prior_like = DiagonalGaussianBelief(torch.zeros(1, 8), torch.zeros(1, 8))  # exactly N(0, I)
    for k in (1, 2, 4, 8):
        fused = DiagonalGaussianBelief.fuse([prior_like] * k)
        torch.testing.assert_close(fused.logvar, torch.zeros(1, 8), atol=1e-5, rtol=0)


def test_fusion_weights_by_precision_not_uniformly() -> None:
    """A confident view must dominate an uncertain one. Averaging the means would not."""
    confident = DiagonalGaussianBelief(torch.full((1, 2), 5.0), torch.full((1, 2), -4.0))
    vague = DiagonalGaussianBelief(torch.full((1, 2), -5.0), torch.full((1, 2), 2.0))
    fused = DiagonalGaussianBelief.fuse([confident, vague])
    assert torch.all(fused.mu > 3.0), (
        f"fused mean {fused.mu.flatten().tolist()} sits near the vague view; the fusion is not "
        "precision-weighted"
    )


def test_fusing_one_belief_is_the_identity() -> None:
    b = DiagonalGaussianBelief(torch.randn(3, 5), torch.randn(3, 5) * 0.3)
    fused = DiagonalGaussianBelief.fuse([b])
    torch.testing.assert_close(fused.mu, b.mu)
    torch.testing.assert_close(fused.logvar, b.logvar)


def test_fusing_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        DiagonalGaussianBelief.fuse([])


# --------------------------------------------------------------------------------------
# Eq 16 / Eq 17 -- information gain
# --------------------------------------------------------------------------------------


def test_eq17_information_gain_is_ambiguity_reduced() -> None:
    u_before = torch.tensor([0.10, 0.05])
    u_after = torch.tensor([[0.04, 0.09], [0.05, 0.01]])
    ig = information_gain(u_before, u_after)
    torch.testing.assert_close(ig, torch.tensor([[0.06, 0.01], [0.00, 0.04]]))


def test_eq17_a_bad_view_has_NEGATIVE_information_gain() -> None:
    """Negative gain is meaningful and must not be clipped away. Fusing a poor view can make the
    belief WORSE, because it still contributes precision to the fusion. An acquisition net that
    never sees a negative target will happily recommend such a view."""
    ig = information_gain(torch.tensor([0.05]), torch.tensor([[0.12]]))
    assert float(ig) < 0


def test_eq17_rejects_a_batch_reduced_ambiguity() -> None:
    """REGRESSION. The audited `should_sense` meaned U over the BATCH, so one highly ambiguous
    scene among fifteen certain ones could not trigger a look of its own. Sensing is per-scene."""
    with pytest.raises(ValueError, match="per-scene"):
        information_gain(torch.tensor(0.1), torch.rand(4, 8))


def test_eq18_acquisition_loss_is_a_regression_onto_the_true_gain() -> None:
    pred = torch.tensor([[0.1, 0.2]])
    true = torch.tensor([[0.1, 0.2]])
    assert float(acquisition_loss(pred, true)) == pytest.approx(0.0)
    assert float(acquisition_loss(torch.zeros(1, 2), true)) > 0


# --------------------------------------------------------------------------------------
# The deployment path
# --------------------------------------------------------------------------------------


@pytest.fixture
def active():
    torch.manual_seed(0)
    enc = BeliefEncoder(MLPBackbone(16, 64), latent_dim=8)
    acq = AcquisitionNet(64, n_views=8)
    return enc, acq, ActivePerception(enc, acq, ActiveConfig(tau_ambiguity=0.01, sensing_budget=2))


def test_regression_an_unfitted_acquisition_net_refuses_to_steer_the_camera(active) -> None:
    """THE DEFECT THIS BUFFER EXISTS TO PREVENT.

    The audited repository shipped an AcquisitionNet with no loss, no target and no look-ahead,
    and then took an argmax over its RANDOMLY-INITIALIZED output to decide where to point the
    camera. That is an argmax over noise. It must be impossible to do by accident.
    """
    enc, acq, ap = active
    assert not bool(acq.fitted)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        ap.select_view(torch.randn(2, 16), torch.arange(8).unsqueeze(0).expand(2, -1))

    acq.fitted.fill_(True)
    b = ap.select_view(torch.randn(2, 16), torch.arange(8).unsqueeze(0).expand(2, -1))
    assert b.shape == (2,), "select_view must return one viewpoint PER SCENE"


def test_regression_view_selection_argmaxes_over_the_viewpoint_axis(active) -> None:
    """REGRESSION -- THE ALWAYS-VIEW-ZERO BUG.

    The audited selector called `argmax(dim=-1)` on a (B, Nb, 1) tensor. That reduces over the
    trailing SIZE-1 axis, so it returned a constant 0 of shape (B, Nb) -- active perception chose
    viewpoint 0 for every scene, forever, and nobody noticed because 0 is a plausible answer.

    Asserting "the chosen views vary" against an UNTRAINED net does not test this: a randomly
    initialized net legitimately prefers one viewpoint for every input, because the view embedding
    swamps the observation features. So we install known scores and check the selector finds the
    argmax where we put it.
    """
    enc, _, _ = active

    class _KnownScores(AcquisitionNet):
        """alpha_omega whose argmax is at a known, deliberately non-zero index."""

        def forward(self, obs_features, view_ids):  # type: ignore[override]
            b, nb = view_ids.shape
            s = torch.zeros(b, nb)
            best = torch.arange(b) % nb  # scene i prefers view i mod nb
            s[torch.arange(b), best] = 1.0
            return s

    acq = _KnownScores(64, n_views=8)
    acq.fitted.fill_(True)
    ap = ActivePerception(enc, acq, ActiveConfig())

    b = 16
    chosen = ap.select_view(torch.randn(b, 16), torch.arange(8).unsqueeze(0).expand(b, -1))
    assert chosen.shape == (b,), "one viewpoint PER SCENE, not (B, Nb)"
    torch.testing.assert_close(chosen, torch.arange(b) % 8)


def test_sensing_is_a_per_scene_decision(active) -> None:
    _, _, ap = active
    u = torch.tensor([0.0, 0.5, 0.0])  # only the middle scene is ambiguous
    torch.testing.assert_close(
        ap.should_sense(u, views_taken=0), torch.tensor([False, True, False])
    )


def test_the_sensing_budget_is_enforced(active) -> None:
    """Alg 2: 'if U > tau_U AND SENSING BUDGET REMAINS'. Without the budget, a deployed loop that
    stays ambiguous senses forever."""
    _, _, ap = active
    u = torch.full((3,), 10.0)  # maximally ambiguous
    assert bool(ap.should_sense(u, views_taken=0).all())
    assert bool(ap.should_sense(u, views_taken=1).all())
    assert not bool(ap.should_sense(u, views_taken=2).any()), "budget of 2 was exceeded"


# --------------------------------------------------------------------------------------
# The winner's curse -- the trap that inflates C3's headline
# --------------------------------------------------------------------------------------


def test_max_over_noisy_estimates_is_biased_upward() -> None:
    """THE STATISTICAL TRAP, in isolation.

    IG(b) is a Monte Carlo estimate, so taking a max over |B| of them partly selects for whichever
    estimate happened to be noisiest -- the winner's curse. The apparent gain then includes a
    measurement of your own sampling error.

    Here the TRUE gain is identically zero for every viewpoint. A naive max-of-N still reports a
    large positive gain. Selecting on one estimate and scoring on an independent one reports ~0,
    which is the truth.

    Measured on the real RGB-D corpus this was not a subtlety: 25.4 of an apparent 43 points of
    ambiguity reduction were pure selection noise.
    """
    torch.manual_seed(0)
    b, nb = 4096, 8
    true_gain = torch.zeros(b, nb)  # no viewpoint is actually any good

    est_a = true_gain + 0.1 * torch.randn(b, nb)
    est_b = true_gain + 0.1 * torch.randn(b, nb)  # independent estimate of the SAME thing

    naive = est_a.max(dim=1).values.mean()  # select AND score on est_a
    honest = est_b.gather(1, est_a.argmax(dim=1, keepdim=True)).mean()  # select on a, score on b

    assert float(naive) > 0.1, "max-of-8 on pure noise should look strongly positive -- that's the bug"
    assert abs(float(honest)) < 0.01, "the bias-corrected estimate must recover the truth (zero)"


def test_evaluate_active_perception_reports_the_bias(monkeypatch) -> None:
    """The report must expose how much of the apparent gain was selection noise, so a paper cannot
    quote the flattering number by accident."""
    from msp.diagnostics.active_eval import ActiveReport

    r = ActiveReport(
        u_one_view=0.004, reduction_random=0.146, reduction_oracle_biased=0.430,
        reduction_honest=0.175, reduction_learned=None, selection_bias=0.255,
        n_scenes=512, n_views=8,
    )
    assert r.selection_bias == pytest.approx(r.reduction_oracle_biased - r.reduction_honest)
    assert "report this" in r.summary()
    assert "Do not report it" in r.summary()
