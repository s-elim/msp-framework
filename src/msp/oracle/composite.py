"""The three-tier outcome operator. Formalization Section 11, and V3 Change 1.

    tier 1  AnalyticGraspOracle   differentiable Ferrari-Canny prior   (fast, everywhere)
    tier 2  MuJoCoOracle          rigid-body contact rollouts          (slow, ground truth)
    tier 3  residual on REAL grasp outcomes                            (NOT BUILT)

WHAT EACH TIER SUPPLIES, AND WHY.

  Phi(x)   -- from tier 1, always.
             J(x) = dPhi/dx is obtained by autodiff (Section 11). Tier 2 is not differentiable
             and finite-differencing a contact simulator produces noise, not a Jacobian.

  margin   -- from tier 1.
             The wrench-space stability margin IS the Ferrari-Canny epsilon. MuJoCo has no such
             metric, and inventing one from contact forces would be a different quantity wearing
             the same name.

  success  -- from tier 2 when it is available, else tier 1.
  slip     -- from tier 2 when it is available, else tier 1.
             These are DYNAMIC events. The analytic tier is quasi-static: it cannot see the object
             topple as the jaws close, roll out of the grasp, or slip during the lift. Those are
             precisely the contact-rich failures the T-RO reviewers care about.

THE TIER GAP IS A RESULT, NOT AN IMPLEMENTATION DETAIL.

`tier_gap()` measures how much tiers 1 and 2 disagree on the same (x, a). That number is the
in-simulation analogue of the sim-to-real sufficiency gap which the review calls "the paper's
decisive experiment" -- it says how much of the outcome the cheap differentiable prior is getting
wrong. Reporting it is what makes the composition honest rather than a convenience.

WHAT IS STILL MISSING. Tier 3 -- a low-parameter residual fit to a few thousand REAL robot grasp
outcomes -- does not exist, and it is the tier the reviewers say carries the paper. Until it does,
this operator is grounded in simulated physics only, and no claim here transfers to hardware.
`RealResidualOracle` is deliberately absent rather than stubbed, because a stub would let someone
believe the grounding exists.
"""

from __future__ import annotations

import torch
from torch import Tensor

from msp.oracle.analytic import AnalyticGraspOracle
from msp.oracle.base import PhysicsOracle
from msp.types import Outcome

__all__ = ["CompositeOracle"]


class CompositeOracle(PhysicsOracle):
    """Analytic prior composed with a rigid-body simulator.

    Args:
        analytic: tier 1. Supplies Phi, J(x), and the margin. Required.
        simulator: tier 2. Supplies success and slip. Optional -- without it the operator is the
            analytic tier alone, which is fine for fast iteration and NOT fine for a paper.
    """

    differentiable = True  # because Phi comes from tier 1

    def __init__(
        self,
        analytic: AnalyticGraspOracle,
        simulator: PhysicsOracle | None = None,
    ) -> None:
        self.analytic = analytic
        self.simulator = simulator

    @property
    def state_dim(self) -> int:
        return self.analytic.state_dim

    def to(self, device: torch.device | str) -> CompositeOracle:
        self.analytic.to(device)
        return self

    # -- Phi: always tier 1, always differentiable ----------------------------

    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:
        """Phi(x) from the analytic tier.

        NOTE FOR THE PAPER. This means the identifiability result (Theorem 4, contribution C2) is
        a statement about the ANALYTIC outcome map. Whether ker J(x) computed here is also
        invariant under the simulator is an empirical question, and `tier_gap` is how you answer
        it. Do not report C2 without also reporting the gap.
        """
        return self.analytic.outcome_params(state, actions)

    # -- y: dynamic quantities from tier 2 when present -----------------------

    @torch.no_grad()
    def query(self, state: Tensor, actions: Tensor) -> Outcome:
        analytic = self.analytic.query(state, actions)
        if self.simulator is None:
            return analytic

        sim = self.simulator.query(state, actions)
        out = Outcome(
            succ=sim.succ,  # the object really did leave the table
            margin=analytic.margin,  # the wrench-space quality; MuJoCo has no such metric
            slip=sim.slip,  # the real post-lift deviation
        )
        out.validate()
        return out

    # -- the gap between the tiers IS a reportable number ---------------------

    @torch.no_grad()
    def tier_gap(self, state: Tensor, actions: Tensor) -> dict[str, float]:
        """How much the cheap differentiable prior disagrees with the simulator.

        This is the in-simulation analogue of the sim-to-real sufficiency gap. A large gap means
        the analytic tier -- which is what the encoder is trained against for the bulk of the
        corpus -- is systematically wrong, and the identifiability result computed from its Phi
        is a statement about a fiction.

        Returns agreement, the analytic and simulated success rates, and the slip correlation.
        """
        if self.simulator is None:
            raise RuntimeError("tier_gap needs a simulator; this composite has only tier 1.")

        a = self.analytic.query(state, actions)
        s = self.simulator.query(state, actions)

        a_succ = a.succ.flatten()
        s_succ = s.succ.flatten()
        agree = float((a_succ == s_succ).float().mean())

        a_slip = a.slip.flatten()
        s_slip = s.slip.flatten()
        if a_slip.std() > 1e-8 and s_slip.std() > 1e-8:
            corr = float(
                ((a_slip - a_slip.mean()) * (s_slip - s_slip.mean())).mean()
                / (a_slip.std() * s_slip.std())
            )
        else:
            corr = float("nan")

        return {
            "success_agreement": agree,
            "analytic_success_rate": float(a_succ.mean()),
            "simulated_success_rate": float(s_succ.mean()),
            # False positives are the dangerous direction: the analytic prior says "grasp is
            # good", the simulator says the object fell. Those are the labels that teach the
            # encoder to be confidently wrong.
            "analytic_false_positive_rate": float(
                ((a_succ == 1) & (s_succ == 0)).float().sum() / (a_succ == 1).float().sum().clamp_min(1)
            ),
            "analytic_false_negative_rate": float(
                ((a_succ == 0) & (s_succ == 1)).float().sum() / (a_succ == 0).float().sum().clamp_min(1)
            ),
            "slip_correlation": corr,
            "n": int(a_succ.numel()),
        }

    # -- scene / action sampling delegates to tier 1 --------------------------

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        return self.analytic.sample_states(n, generator=generator)

    def sample_actions(
        self, state: Tensor, n_actions: int, generator: torch.Generator | None = None
    ) -> Tensor:
        return self.analytic.sample_actions(state, n_actions, generator=generator)

    def observe(self, state: Tensor, obs_dim: int = 64, noise: float = 0.01) -> Tensor:
        return self.analytic.observe(state, obs_dim=obs_dim, noise=noise)
