"""Test-time adaptation: assimilate a probe outcome into the belief. Eq 19-21.

THREE DEFECTS THIS MODULE EXISTS TO NOT HAVE. All three were in the audited version and all
three are pinned by regression tests.

1. IT CRASHED. The old loop read the encoder's live mu_0/logvar_0 inside its KL term, so
   `backward()` freed the encoder graph on step 1 and step 2 raised
   `RuntimeError: Trying to backward through the graph a second time`. TTA with steps > 1
   had never been executed. Fixed by `Belief.as_free_parameters()`, which returns LEAVES.

2. THE VARIANCE NEVER MOVED. The old loop evaluated the head at the belief's MEAN, so
   d(NLL)/d(logvar) was identically zero. Since epistemic variance v is what active
   perception consumes, TTA could not reduce the ambiguity U that contribution C3 claims it
   reduces -- the two halves of C3 were not the same operation, one was a no-op. Fixed by
   reparameterized sampling inside the loop.

3. THE FIXED POINT WAS WRONG. The old loop minimized `nll + trust_region * KL` with
   trust_region = 0.1. The stationary point of  min E_q[-log p] + lam*KL(q||q0)  is
   q* ∝ q0 * p^(1/lam), so lam = 0.1 converges to q0 * p^10 -- the single probe likelihood
   raised to the TENTH POWER. Eq 19 is exact Bayes and requires lam = 1. The parameter named
   "trust region" was in fact AMPLIFYING the update tenfold, and smaller values made it more
   aggressive, not less. Here the KL coefficient is pinned at 1 and the trust region is what
   its name says: a hard bound on ||mu' - mu||.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.belief import Belief, DiagonalGaussianBelief
from msp.math.divergences import gaussian_nll, log_normal_nll
from msp.models.nets import OutcomeHead
from msp.types import Outcome

__all__ = ["TTAConfig", "adapt_belief"]


@dataclass(frozen=True)
class TTAConfig:
    steps: int = 20
    lr: float = 5e-2
    #: HARD bound on ||mu' - mu||_2, projected after every step. This is the trust region of
    #: Eq 20 ("k gradient steps with a trust region, bounded ||mu' - mu||"). It is a
    #: CONSTRAINT, not a penalty weight -- see defect 3 above.
    trust_radius: float = 1.0
    #: Coefficient on KL(q' || q_theta). Eq 19/20 require EXACTLY 1.0 for the fixed point to
    #: be the Bayes posterior. Exposed only so ablations can deliberately temper it; changing
    #: it means you are no longer computing Eq 19.
    kl_weight: float = 1.0
    #: Posterior samples used to estimate E_{z~q'}[-log p]. Must be >= 2; more is a lower-
    #: variance gradient at linear cost.
    num_samples: int = 32


def adapt_belief(
    belief: Belief,
    head: OutcomeHead,
    probe_action: Tensor,
    probe_outcome: Outcome,
    config: TTAConfig | None = None,
    gripper: Tensor | None = None,
) -> DiagonalGaussianBelief:
    """Eq 19/20: q'(z) ∝ q_theta(z|o) * p_psi(y_p | z, a_p), realized variationally.

        min_{mu', sigma'}  E_{z~q'}[ -log p_psi(y_p | z, a_p) ]  +  KL( q' || q_theta(z|o) )

    The FULL outcome likelihood is assimilated -- success, margin AND slip. The audited
    version used only a binary cross-entropy on success and discarded the observed margin
    and slip, throwing away two thirds of the information the probe returned.

    For a scene-PERTURBING probe (Eq 21), the caller re-encodes from a fresh observation o'
    through the known transition Tk and passes the resulting belief as `belief`. The
    optimization is identical; only the prior changes. That is exactly what Eq 21 says.

    Args:
        belief:        q_theta(z | o), the prior. Detached internally; not mutated.
        head:          p_psi. FROZEN -- gradients do not reach psi.
        probe_action:  (B, 1, Ad) the action actually executed.
        probe_outcome: the realized y_p, fields (B, 1, 1).

    Returns:
        The adapted posterior q'.
    """
    cfg = config or TTAConfig()
    if not isinstance(belief, DiagonalGaussianBelief):
        raise TypeError(f"TTA currently supports DiagonalGaussianBelief; got {type(belief)}")
    if cfg.num_samples < 2:
        raise ValueError("num_samples must be >= 2 to estimate E_{z~q'} with any variance")

    prior = belief.detach()
    mu, logvar = prior.as_free_parameters()  # LEAF tensors -- see defect 1
    mu0 = prior.mu.clone()

    # Freeze psi. TTA adapts the BELIEF, not the network. The audited version left the head
    # unfrozen, so head.grad accumulated across every TTA step and persisted (measured
    # ||head.grad||_1 = 252) -- a later optimizer.step() would have applied probe-adaptation
    # gradients to psi.
    head_was_training = head.training
    head.eval()
    saved = [p.requires_grad for p in head.parameters()]
    for p in head.parameters():
        p.requires_grad_(False)

    try:
        opt = torch.optim.Adam([mu, logvar], lr=cfg.lr)
        for _ in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            q = DiagonalGaussianBelief(mu, logvar)

            # E_{z~q'}[ -log p_psi(y_p | z, a_p) ] via reparameterized samples.
            # Sampling is what couples logvar to the likelihood -- see defect 2.
            z = q.rsample(cfg.num_samples)  # (B, K, d)
            pred = head(z, probe_action, gripper=gripper)  # (B, K, 1, 1)

            y = probe_outcome
            nll = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    pred.succ_logit, y.succ.unsqueeze(1).expand_as(pred.succ_logit),
                    reduction="none",
                )
                + gaussian_nll(
                    y.margin.unsqueeze(1).expand_as(pred.margin_mu),
                    pred.margin_mu, pred.margin_logvar,
                )
                + log_normal_nll(
                    y.slip.unsqueeze(1).expand_as(pred.slip_log_mu),
                    pred.slip_log_mu, pred.slip_log_logvar,
                )
            ).mean(dim=1).sum()  # mean over K (the expectation), sum over batch

            kl = q.kl_to(prior).sum()
            (nll + cfg.kl_weight * kl).backward()
            opt.step()

            # Project onto the trust region: a HARD bound on the mean's displacement.
            with torch.no_grad():
                delta = mu - mu0
                norm = delta.norm(dim=-1, keepdim=True)
                scale = (cfg.trust_radius / norm.clamp_min(1e-12)).clamp(max=1.0)
                mu.copy_(mu0 + delta * scale)
    finally:
        for p, r in zip(head.parameters(), saved, strict=True):
            p.requires_grad_(r)
        if head_was_training:
            head.train()

    return DiagonalGaussianBelief(mu.detach(), logvar.detach())
