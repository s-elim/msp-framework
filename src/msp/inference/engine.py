"""Algorithm 2: perceive, decide, sense, adapt, or ABSTAIN.

The whole of MSP's deployment behaviour is derived here from the two learned modules. No new
network, no policy, no RL -- exactly as Step 5 of the blueprint promises.

The single most important line in this file is that `select` returns `Decision`, a sum type
of `ActionChoice | Abstain`. The audited engine returned an integer index; when the certified
set was empty it filled every score with -inf and `argmax` dutifully returned index 0,
silently executing an UNCERTIFIED grasp. Abstention -- contribution C4 and the entire safety
story of the paper -- was not merely unimplemented, it failed in the unsafe direction. A sum
type makes that unrepresentable: the caller must handle `Abstain` or fail to typecheck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.belief import Belief
from msp.inference.calibrator import ConformalCalibrator
from msp.math.decision import SuccessStats, ambiguity, risk_averse_score, success_stats
from msp.models.nets import BeliefEncoder, OutcomeHead
from msp.types import ABSTAIN, ActionChoice, Decision

__all__ = ["InferenceConfig", "ScoredActions", "InferenceEngine"]


@dataclass(frozen=True)
class InferenceConfig:
    num_samples: int = 32  # K. Section 11 prescribes 8-32.
    lambda_risk: float = 1.0  # Eq 15
    tau_ambiguity: float = 0.05  # tau_U, Eq 16 -- the sensing trigger
    sensing_budget: int = 2  # Alg 2: "and sensing budget remains"


@dataclass(frozen=True)
class ScoredActions:
    """Everything Algorithm 2 needs, computed once per observation."""

    stats: SuccessStats  # s, v  -- Eq 13, 14
    ambiguity: Tensor  # U      -- Eq 16, per scene (B,)
    scores: Tensor  # s - lambda*v -- Eq 15
    actions: Tensor  # (B, Na, Ad)


class InferenceEngine:
    """Stateless orchestrator over (encoder, head, calibrator)."""

    def __init__(
        self,
        encoder: BeliefEncoder,
        head: OutcomeHead,
        calibrator: ConformalCalibrator | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.head = head
        self.calibrator = calibrator
        self.cfg = config or InferenceConfig()

    # -- perception ----------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        belief: Belief,
        actions: Tensor,
        gripper: Tensor | None = None,
    ) -> ScoredActions:
        """Marginalize the belief through the head. Eq 13, 14, 15, 16."""
        probs = belief.expected_success(
            self.head, actions, self.cfg.num_samples, gripper=gripper
        )  # (B, K, Na)
        stats = success_stats(probs)
        return ScoredActions(
            stats=stats,
            ambiguity=ambiguity(stats),
            scores=risk_averse_score(stats, self.cfg.lambda_risk),
            actions=actions,
        )

    @torch.no_grad()
    def perceive(self, obs: Tensor) -> Belief:
        return self.encoder(obs)

    # -- decision ------------------------------------------------------------

    def select(self, scored: ScoredActions, batch_index: int = 0) -> Decision:
        """Eq 24 then Eq 15: certify, then choose -- and ABSTAIN if nothing certifies.

        The ordering is not cosmetic. Selecting first and certifying second would let a
        high-scoring uncertified action leak through; certifying first makes the certificate
        a precondition of the argmax, which is what Eq 24 states.
        """
        s = scored.stats.s[batch_index]  # (Na,)
        v = scored.stats.v[batch_index]
        scores = scored.scores[batch_index]

        if self.calibrator is None:
            raise RuntimeError(
                "InferenceEngine has no calibrator. MSP does not act without a certificate: "
                "fit a ConformalCalibrator on a held-out fold, or use `select_uncertified` "
                "and accept that you have left the framework's guarantee behind."
            )

        certified = self.calibrator.certified_mask(s.unsqueeze(0))[0]  # (Na,) bool

        if not bool(certified.any()):
            return ABSTAIN  # Eq 24. The whole point.

        masked = scores.masked_fill(~certified, float("-inf"))
        idx = int(masked.argmax().item())
        return ActionChoice(
            index=idx,
            action=scored.actions[batch_index, idx].clone(),
            success_prob=float(s[idx]),
            epistemic_var=float(v[idx]),
            score=float(scores[idx]),
        )

    def select_uncertified(self, scored: ScoredActions, batch_index: int = 0) -> ActionChoice:
        """Eq 15 with NO certificate. For ablations and for baselines that have no notion of
        abstention -- never for a deployed run. Named to be conspicuous in a diff."""
        scores = scored.scores[batch_index]
        idx = int(scores.argmax().item())
        return ActionChoice(
            index=idx,
            action=scored.actions[batch_index, idx].clone(),
            success_prob=float(scored.stats.s[batch_index, idx]),
            epistemic_var=float(scored.stats.v[batch_index, idx]),
            score=float(scores[idx]),
        )

    # -- sensing trigger -----------------------------------------------------

    def should_sense(self, scored: ScoredActions) -> Tensor:
        """U(o) > tau_U, PER SCENE. (B,) bool.

        Returned per scene, never reduced to one bool for the batch. The audited version
        meaned over the batch, so a single highly ambiguous scene could not trigger sensing
        on its own.
        """
        return scored.ambiguity > self.cfg.tau_ambiguity
