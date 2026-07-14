"""The Belief abstraction: reparameterization, divergences, and graph hygiene.

The regression tests here pin the two defects that made the audited TTA both wrong and
unrunnable.
"""

from __future__ import annotations

import pytest
import torch

from msp.belief import DiagonalGaussianBelief
from msp.math.divergences import diagonal_gaussian_kl


def test_rsample_shape_and_reparameterization() -> None:
    mu = torch.randn(4, 8, requires_grad=True)
    logvar = torch.zeros(4, 8, requires_grad=True)
    z = DiagonalGaussianBelief(mu, logvar).rsample(16)

    assert z.shape == (4, 16, 8)
    z.sum().backward()
    assert mu.grad is not None and torch.any(mu.grad != 0)


def test_regression_rsample_gives_logvar_a_gradient() -> None:
    """REGRESSION -- THE FROZEN-SIGMA DEFECT.

    The audited TTA evaluated the outcome head at the belief's MEAN, so d(NLL)/d(logvar) was
    identically zero and the belief's variance never moved (measured ||dlogvar|| = 0.0 after
    20 steps). Since epistemic variance v is what active perception consumes, TTA could not
    reduce the ambiguity U that contribution C3 says it reduces.

    Sampling is what couples logvar to any downstream likelihood. `Belief.rsample` is the
    only route to the head, so the bug is no longer expressible -- but we pin it anyway.
    """
    mu = torch.zeros(2, 4, requires_grad=True)
    logvar = torch.zeros(2, 4, requires_grad=True)
    belief = DiagonalGaussianBelief(mu, logvar)

    # Any downstream scalar computed from samples must produce logvar gradient.
    z = belief.rsample(32)
    (z**2).sum().backward()

    assert logvar.grad is not None
    assert torch.any(logvar.grad != 0), (
        "logvar received no gradient from a sample-based loss. The reparameterization is "
        "broken and the belief's variance can never adapt."
    )


def test_rsample_is_statistically_correct() -> None:
    """Empirical moments of the sampler must match (mu, sigma^2)."""
    torch.manual_seed(0)
    mu = torch.tensor([[1.0, -2.0]])
    logvar = torch.log(torch.tensor([[4.0, 0.25]]))  # sigma = 2.0, 0.5
    z = DiagonalGaussianBelief(mu, logvar).rsample(200_000)

    torch.testing.assert_close(z.mean(dim=1), mu, atol=0.02, rtol=0)
    torch.testing.assert_close(
        z.std(dim=1), torch.tensor([[2.0, 0.5]]), atol=0.02, rtol=0
    )


def test_kl_to_self_is_zero() -> None:
    """KL(q || q) = 0. Catches sign and direction slips in the trust region."""
    b = DiagonalGaussianBelief(torch.randn(5, 8), torch.randn(5, 8) * 0.3)
    torch.testing.assert_close(b.kl_to(b), torch.zeros(5), atol=1e-5, rtol=1e-5)


def test_kl_to_prior_is_the_special_case_of_kl_to() -> None:
    """kl_to_prior() must equal kl_to(N(0, I)). One implementation, two entry points --
    the audited code had two hand-written implementations that drifted apart."""
    b = DiagonalGaussianBelief(torch.randn(5, 8), torch.randn(5, 8) * 0.3)
    prior = DiagonalGaussianBelief(torch.zeros(5, 8), torch.zeros(5, 8))
    torch.testing.assert_close(b.kl_to_prior(), b.kl_to(prior), atol=1e-5, rtol=1e-5)


def test_kl_direction_is_not_symmetric() -> None:
    """KL is not a metric. Pin the direction so a future edit cannot silently swap the
    arguments of the TTA trust region KL(q' || q_theta)."""
    q = DiagonalGaussianBelief(torch.ones(1, 4), torch.zeros(1, 4))
    p = DiagonalGaussianBelief(torch.zeros(1, 4), torch.ones(1, 4))
    assert not torch.allclose(q.kl_to(p), p.kl_to(q))
    torch.testing.assert_close(
        q.kl_to(p), diagonal_gaussian_kl(q.mu, q.logvar, p.mu, p.logvar)
    )


def test_regression_as_free_parameters_returns_leaves() -> None:
    """REGRESSION -- THE TTA CRASH.

    The audited TTA read the encoder's live mu_0/logvar_0 inside its KL term. backward()
    freed that graph on step 1, so step 2 raised
    `RuntimeError: Trying to backward through the graph a second time`. TTA had never been
    executed. `as_free_parameters` must return graph-free LEAF tensors so the TTA loop can
    call backward() repeatedly.
    """
    encoder_out_mu = torch.randn(2, 4, requires_grad=True) * 2  # non-leaf: has graph history
    encoder_out_lv = torch.randn(2, 4, requires_grad=True) * 2
    assert not encoder_out_mu.is_leaf

    belief = DiagonalGaussianBelief(encoder_out_mu, encoder_out_lv)
    mu, logvar = belief.as_free_parameters()

    assert mu.is_leaf and logvar.is_leaf
    assert mu.requires_grad and logvar.requires_grad
    assert mu.grad_fn is None and logvar.grad_fn is None

    # Repeated backward() through these leaves must not raise.
    for _ in range(5):
        loss = (mu**2).sum() + (logvar**2).sum()
        loss.backward()


def test_detach_severs_the_graph() -> None:
    mu = torch.randn(2, 4, requires_grad=True)
    b = DiagonalGaussianBelief(mu, torch.zeros(2, 4))
    d = b.detach()
    assert not d.mu.requires_grad
    assert b.mu.requires_grad, "detach() must not mutate the original belief"


def test_logvar_is_clamped_on_construction() -> None:
    """exp(0.5*logvar) must not overflow; exp(-logvar) must not overflow downstream."""
    b = DiagonalGaussianBelief(torch.zeros(1, 3), torch.tensor([[-1e4, 0.0, 1e4]]))
    assert torch.all(b.logvar >= DiagonalGaussianBelief.LOGVAR_MIN)
    assert torch.all(b.logvar <= DiagonalGaussianBelief.LOGVAR_MAX)
    assert torch.all(torch.isfinite(b.rsample(4)))


def test_shape_contract_is_enforced() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        DiagonalGaussianBelief(torch.zeros(2, 4), torch.zeros(2, 5))
    with pytest.raises(ValueError, match=r"expected \(B, d\)"):
        DiagonalGaussianBelief(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))


def test_no_mean_shortcut_on_the_interface() -> None:
    """`Belief` deliberately exposes no `.mean()`. Evaluating the head at the mean is the
    bug this abstraction exists to prevent; a point estimate must be requested explicitly
    via `point_estimate()`, which is greppable in review."""
    assert not hasattr(DiagonalGaussianBelief, "mean")
    assert hasattr(DiagonalGaussianBelief, "point_estimate")
