"""Core domain types for MSP.

These types encode the objects of the formalization (Section 0) directly. They exist
so that the scientific invariants of the framework are enforced by the type system
rather than by convention.

Two invariants are load-bearing and are the reason several of these types exist at all:

* An outcome is a *distribution*, never a point estimate. `OutcomeDistribution` carries
  the parameters of p_psi(y | z, a) and refuses to be confused with a realized `Outcome`.
* A decision is either an action or an abstention. `Decision` is a sum type, so a caller
  cannot silently execute an uncertified action when the certified set is empty (Eq 24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

# Semantic aliases. These document tensor rank at the call site; they are not enforced.
# Shape conventions used throughout the codebase:
#   B   batch of scenes
#   K   posterior samples drawn from a Belief
#   Na  candidate actions per scene
#   d   latent dimension of the sufficient statistic z
#   Ad  action dimension (7 = translation(3) + unit quaternion(4))
Observation: TypeAlias = Tensor  # (B, 4, H, W) RGB-D
Latent: TypeAlias = Tensor  # (B, d) or (B, K, d)
ActionTensor: TypeAlias = Tensor  # (B, Na, Ad)
GripperTensor: TypeAlias = Tensor  # (B, Gd) embodiment descriptor g

ACTION_DIM: int = 7
OUTCOME_DIM: int = 3  # (succ, margin, slip)


@dataclass(frozen=True)
class Outcome:
    """A realized outcome y = (succ, margin, slip), the codomain of the physics kernel M.

    Formalization Section 0::

        Y = {0,1} x R x R_{>=0}

    `slip` is non-negative by definition. We do not clamp it here (that would hide oracle
    bugs); `validate` is provided for callers that want the assertion.

    All fields share shape (..., 1) so they broadcast against outcome-head predictions.
    """

    succ: Tensor
    margin: Tensor
    slip: Tensor

    def validate(self) -> None:
        """Raise if the outcome violates the domain of Y. Cheap; call it in oracles."""
        if not torch.all((self.succ == 0) | (self.succ == 1)):
            raise ValueError("Outcome.succ must be binary; got values outside {0, 1}.")
        if torch.any(self.slip < 0):
            raise ValueError("Outcome.slip must be non-negative (Y = {0,1} x R x R_>=0).")

    def to(self, device: torch.device | str) -> Outcome:
        return Outcome(self.succ.to(device), self.margin.to(device), self.slip.to(device))

    def expand_to(self, k: int) -> Outcome:
        """Broadcast a realized (B, Na, 1) outcome against a K-sample axis -> (B, K, Na, 1).

        The realized outcome does not depend on which posterior sample z_k produced the
        prediction, so it is simply repeated. Needed to average the distortion over K draws
        when estimating a rate-distortion frontier point.
        """
        return Outcome(
            succ=self.succ.unsqueeze(1).expand(-1, k, -1, -1),
            margin=self.margin.unsqueeze(1).expand(-1, k, -1, -1),
            slip=self.slip.unsqueeze(1).expand(-1, k, -1, -1),
        )

    @property
    def shape(self) -> torch.Size:
        return self.succ.shape


def _bounded_logvar(raw: Tensor, var_floor: float = 0.02, logvar_max: float = 6.0) -> Tensor:
    """Map a raw network output to a log-variance with a hard FLOOR on the variance.

        logvar = log( var_floor + softplus(raw) ),   then capped above.

    TWO BUGS THIS REPLACES, and the second one is the expensive one.

    (1) A raw `clamp` has ZERO GRADIENT outside its range. A head that saturates the bound stops
        receiving any signal to come back, and is stuck there for the rest of training. Softplus
        is monotone and differentiable everywhere, so the bound is respected without a dead zone.

    (2) THE FLOOR IS THE POINT. With the old clamp at logvar >= -10, the head could claim a
        precision of exp(10) ~ 22,000. It duly overfit the slip magnitude on the training fold,
        was confidently wrong on held-out data, and 0.5 * 22000 * (log y - mu)^2 detonated:
        measured on the RGB-D corpus, slip distortion was -1.06 on train and +1697 on validation,
        which was 1697 of the 1697 total. The entire reported rate-distortion frontier was one
        term blowing up.

        Zero-inflating the likelihood was necessary but not sufficient -- it fixed the mass at
        zero and left the positive part free to be arbitrarily overconfident. A model of a noisy
        physical quantity has no business claiming a precision of 22,000, and a floor says so.
        var_floor = 0.02 caps the precision at 50.
    """
    return torch.log(var_floor + torch.nn.functional.softplus(raw)).clamp(max=logvar_max)


@dataclass(frozen=True)
class OutcomeDistribution:
    """Parameters of the learned outcome kernel p_psi(y | z, a).

    Formalization Eq 10 / Section 3::

        p_psi(y|z,a) = Bern(succ; sigmoid(f_psi))
                     * N(margin; m_psi, s_psi^2)
                     * LogNormal(slip; ...)

    Deliberate choices, each fixing an audited defect in the previous implementation:

    * `succ_logit` is a *logit*, never a probability. Downstream code must go through
      `success_prob()`, which uses a numerically stable sigmoid, so no call site can
      accidentally apply sigmoid twice or take log of a probability.
    * `margin_logvar` / `slip_logvar` are clamped on construction. The previous code left
      them unclamped, so `exp(-logvar)` in the Gaussian NLL overflowed to inf/NaN once the
      head grew confident.
    * `slip` is modeled in log-space, because Y constrains slip >= 0 and an unconstrained
      Gaussian places mass on physically impossible negative slip.
    """

    succ_logit: Tensor
    margin_mu: Tensor
    margin_logvar: Tensor
    #: P(the grasp slipped at all). Slip is ZERO-INFLATED: a grasp that holds slips exactly zero,
    #: a grasp that fails slips a lot, and there is very little in between. Forcing a unimodal
    #: log-normal onto that bimodal truth is not a modelling nicety -- it detonates. Measured on
    #: the real RGB-D corpus: train slip distortion -1.66, validation +965.2, and the validation
    #: number was 966 of the total 966, so the entire rate-distortion frontier was just the slip
    #: term exploding. A confident-but-wrong log-normal makes exp(-logvar)*(y-mu)^2 enormous.
    slip_zero_logit: Tensor
    slip_log_mu: Tensor
    slip_log_logvar: Tensor

    #: Variance FLOOR, not a clamp. See `_bounded_logvar`.
    VAR_FLOOR: float = 0.02  # sigma >= 0.14, so exp(-logvar) <= 50
    LOGVAR_MAX: float = 6.0

    def __post_init__(self) -> None:
        # frozen dataclass: mutate through object.__setattr__
        object.__setattr__(self, "margin_logvar", _bounded_logvar(self.margin_logvar))
        object.__setattr__(self, "slip_log_logvar", _bounded_logvar(self.slip_log_logvar))

    def success_prob(self) -> Tensor:
        """sigma_psi(z, a) = sigmoid(f_psi(z, a)). Formalization Eq 13."""
        return torch.sigmoid(self.succ_logit)

    def float(self) -> OutcomeDistribution:
        """Upcast every parameter to fp32.

        Needed because the network may run under bf16 autocast while the LOSS must not.
        bf16 carries ~8 mantissa bits (2-3 decimal digits), and the rate and distortion are
        not merely training signals here -- they are the coordinates plotted on the paper's
        rate-distortion frontier. Training in bf16 is correct; *measuring* in bf16 is not.
        """
        return OutcomeDistribution(
            succ_logit=self.succ_logit.float(),
            margin_mu=self.margin_mu.float(),
            margin_logvar=self.margin_logvar.float(),
            slip_zero_logit=self.slip_zero_logit.float(),
            slip_log_mu=self.slip_log_mu.float(),
            slip_log_logvar=self.slip_log_logvar.float(),
        )

    def expand_to(self, k: int) -> OutcomeDistribution:
        """Broadcast a (B, Na, 1) distribution against a K-sample axis -> (B, K, Na, 1)."""
        return OutcomeDistribution(
            succ_logit=self.succ_logit.unsqueeze(1).expand(-1, k, -1, -1),
            margin_mu=self.margin_mu.unsqueeze(1).expand(-1, k, -1, -1),
            margin_logvar=self.margin_logvar.unsqueeze(1).expand(-1, k, -1, -1),
            slip_zero_logit=self.slip_zero_logit.unsqueeze(1).expand(-1, k, -1, -1),
            slip_log_mu=self.slip_log_mu.unsqueeze(1).expand(-1, k, -1, -1),
            slip_log_logvar=self.slip_log_logvar.unsqueeze(1).expand(-1, k, -1, -1),
        )


class Abstain:
    """The framework declined to act: the certified action set was empty (Eq 24).

    This is a distinct type rather than `None` or index -1 so that a caller cannot
    accidentally treat abstention as an action index. Mishandling it is a type error,
    not a silent uncertified grasp.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Abstain()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Abstain)

    def __hash__(self) -> int:
        return hash(Abstain)


ABSTAIN = Abstain()


@dataclass(frozen=True)
class ActionChoice:
    """A selected action, with the statistics that justified selecting it."""

    index: int
    action: Tensor  # (Ad,)
    success_prob: float  # s(o, a),  Eq 13
    epistemic_var: float  # v(o, a),  Eq 14
    score: float  # s - lambda*v,  Eq 15


Decision: TypeAlias = ActionChoice | Abstain
"""Result of Algorithm 2. Exhaustively matching on this sum type is the only way to
consume an inference result, which is what makes fail-open impossible."""


@dataclass(frozen=True)
class PredictionSet:
    """The conformal prediction set C(o, a) subset of {0, 1}. Formalization Eq 23::

        C(o, a) = { l in {0,1} : score_l(o, a) <= q_hat },
        score_1 = 1 - s,   score_0 = s

    The previous implementation computed only `score_1` and therefore never constructed
    the set at all -- it certified any action whose set was the *ambiguous* {0, 1}. Here
    both memberships are explicit fields, so `is_certified_success` (the singleton test of
    Eq 24) cannot be written incorrectly.

    Fields are boolean tensors of shape (B, Na).
    """

    contains_success: Tensor  # 1 in C(o,a)
    contains_failure: Tensor  # 0 in C(o,a)

    def is_certified_success(self) -> Tensor:
        """C(o, a) == {1}: success is in the set AND failure is not. Eq 24."""
        return self.contains_success & ~self.contains_failure

    def is_empty(self) -> Tensor:
        """C(o, a) == {}: the model is confidently out of distribution."""
        return ~self.contains_success & ~self.contains_failure

    def is_ambiguous(self) -> Tensor:
        """C(o, a) == {0, 1}: both labels plausible. NOT certified."""
        return self.contains_success & self.contains_failure


ProbeKind: TypeAlias = Literal["static", "perturbing"]
"""Whether a probe leaves the scene invariant (Eq 20) or moves it through a known
transition Tk, requiring a re-encode from a fresh observation (Eq 21)."""
