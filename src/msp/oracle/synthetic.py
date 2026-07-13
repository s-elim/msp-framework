"""A synthetic outcome operator with an ANALYTICALLY KNOWN indistinguishability class.

WHY THIS EXISTS -- and why it is arguably the most valuable file in the repository.

Contribution C2 claims that pose and shape are recoverable only up to manipulation-
indistinguishability, and that the residual ambiguity is exactly ker J(x) (Theorem 4). The
simulated T-RO review identifies the experiment that verifies this -- measuring the principal
angles between the predicted null space and the empirically grasp-invariant subspace -- as
the single result that turned Reviewer A from Major Revision to Accept.

That experiment is only trustworthy if the *diagnostic itself* is correct. On a real
simulator you cannot separate "my null-space estimator is broken" from "the theory is
wrong": there is no ground truth to check against.

So we construct a world where the answer is known in closed form.

    Let P be a fixed (rank_out x state_dim) matrix of rank r < d_X.
    Let every outcome parameter depend on x ONLY through the projection  u = P x.

Then Phi(x) = Phi(x') whenever P x = P x', so the indistinguishability class through x is
the affine subspace  x + ker(P),  and

    ker J(x) = ker(P),      dim ker J(x) = state_dim - r,      row J(x) = row(P).

Both are exact, for every x. `SyntheticOracle.true_null_space()` returns an orthonormal
basis for ker(P), and `tests/diagnostics/` asserts that the autodiff-based estimator
recovers it to within numerical tolerance.

This makes Theorem 4 a UNIT TEST rather than an unfalsifiable claim, and it means the
diagnostic is validated before it is ever pointed at a real scene.

The physics is not meant to be realistic. It is meant to have a known answer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from msp.oracle.base import PhysicsOracle
from msp.types import Outcome

__all__ = ["SyntheticOracle"]

#: Parameters emitted per action: (success logit, margin mean, log-slip mean).
PARAMS_PER_ACTION: int = 3


class SyntheticOracle(PhysicsOracle):
    """A smooth, differentiable outcome operator whose null space is known exactly.

    The outcome parameters for action a are::

        u          = P x                             in R^r      (the identifiable part)
        h(x, a)    = tanh( W_a [u ; a] + b_a )       in R^hidden
        succ_logit = w_s . h
        margin     = w_m . h
        log_slip   = w_l . h  - 1.0

    Because x enters only through u = P x, every outcome parameter is invariant to
    perturbations of x inside ker(P). That is the definition of manipulation-
    indistinguishability (Eq 5), realized exactly.

    Args:
        state_dim: d_X.
        rank: r = rank(P). Must be < state_dim, or the null space is trivial and there is
            nothing to identify.
        action_dim: dimension of a.
        seed: fixes P and the random weights, so the world is reproducible.
        noise: Bernoulli/Gaussian sampling noise for `query`. `outcome_params` is always
            noiseless -- J(x) is the Jacobian of the *mean* outcome map, not of a sample.
    """

    differentiable = True

    def __init__(
        self,
        state_dim: int = 6,
        rank: int = 3,
        action_dim: int = 7,
        hidden: int = 16,
        seed: int = 0,
        noise: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0 < rank < state_dim:
            raise ValueError(
                f"rank must satisfy 0 < rank < state_dim to leave a non-trivial null space; "
                f"got rank={rank}, state_dim={state_dim}."
            )
        self._state_dim = state_dim
        self.rank = rank
        self.action_dim = action_dim
        self.noise = noise
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(seed)

        # P: (rank, state_dim), row-orthonormal so that row(P) and ker(P) are clean.
        raw = torch.randn(state_dim, rank, generator=g)
        q, _ = torch.linalg.qr(raw)  # (state_dim, rank), orthonormal columns
        self.P = q.T.contiguous().to(self.device)  # (rank, state_dim)

        in_dim = rank + action_dim
        self.W = (torch.randn(hidden, in_dim, generator=g) / in_dim**0.5).to(self.device)
        self.b = torch.randn(hidden, generator=g).to(self.device)
        self.w_succ = torch.randn(hidden, generator=g).to(self.device) * 2.0
        self.w_margin = torch.randn(hidden, generator=g).to(self.device)
        self.w_slip = torch.randn(hidden, generator=g).to(self.device) * 0.5

    # -- interface -----------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def null_space_dim(self) -> int:
        """dim ker J(x) = state_dim - rank(P). Exact, and the same at every x."""
        return self._state_dim - self.rank

    def true_null_space(self) -> Tensor:
        """Orthonormal basis for ker(P) = ker J(x). Shape (state_dim, state_dim - rank).

        This is the GROUND TRUTH that `msp.diagnostics.identifiability` is validated against.
        """
        # ker(P) is the orthogonal complement of row(P) = col(P^T).
        _, _, vh = torch.linalg.svd(self.P, full_matrices=True)  # vh: (state_dim, state_dim)
        return vh[self.rank :].T.contiguous()  # (state_dim, state_dim - rank)

    def true_row_space(self) -> Tensor:
        """Orthonormal basis for row J(x) -- the IDENTIFIABLE directions. (state_dim, rank)."""
        _, _, vh = torch.linalg.svd(self.P, full_matrices=True)
        return vh[: self.rank].T.contiguous()

    # -- the operator --------------------------------------------------------

    def _params(self, state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (succ_logit, margin, log_slip), each (B, Na, 1). Differentiable in state."""
        b, na, _ = actions.shape
        u = state @ self.P.T  # (B, rank) -- x enters ONLY through this projection
        u_exp = u.unsqueeze(1).expand(b, na, self.rank)  # (B, Na, rank)
        h = torch.tanh(
            torch.cat([u_exp, actions], dim=-1) @ self.W.T + self.b
        )  # (B, Na, hidden)
        succ_logit = (h @ self.w_succ).unsqueeze(-1)
        margin = (h @ self.w_margin).unsqueeze(-1)
        log_slip = (h @ self.w_slip).unsqueeze(-1) - 1.0
        return succ_logit, margin, log_slip

    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:
        """Phi(x) at the given action set. (B, Na * 3). Differentiable in `state`. Eq 25."""
        succ_logit, margin, log_slip = self._params(state, actions)
        stacked = torch.cat([succ_logit, margin, log_slip], dim=-1)  # (B, Na, 3)
        return stacked.reshape(state.shape[0], -1)  # (B, Na*3)

    @torch.no_grad()
    def query(self, state: Tensor, actions: Tensor) -> Outcome:
        """Sample y ~ M(.|x,a)."""
        succ_logit, margin, log_slip = self._params(state, actions)
        p = torch.sigmoid(succ_logit)

        if self.noise:
            succ = torch.bernoulli(p)
            margin = margin + 0.05 * torch.randn_like(margin)
            slip = torch.exp(log_slip + 0.05 * torch.randn_like(log_slip))
        else:
            succ = (p > 0.5).float()
            slip = torch.exp(log_slip)

        # Successful grasps do not slip: couple the coordinates so the outcome vector is
        # physically coherent rather than three independent noises.
        slip = slip * (1.0 - succ) + 1e-4 * succ

        out = Outcome(succ=succ, margin=margin, slip=slip)
        out.validate()
        return out

    def observe(self, state: Tensor, obs_dim: int = 32, noise: float = 0.01) -> Tensor:
        """A stand-in sensor p(o | x): a fixed random linear embedding of x, plus noise.

        Deliberately NOT injective onto the null space in any special way -- the observation
        sees all of x. This matters: the ambiguity that MSP is about is *not* an observation
        limitation, it is a property of the PHYSICS. Even a perfect observation of x cannot
        tell you which point of [x] you are at, because every point of [x] behaves
        identically under every action. Making `observe` fully informative isolates that.
        """
        g = torch.Generator(device="cpu").manual_seed(12345)
        A = (torch.randn(self._state_dim, obs_dim, generator=g) / self._state_dim**0.5).to(
            state.device
        )
        return state @ A + noise * torch.randn(
            state.shape[0], obs_dim, device=state.device
        )

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw x ~ p(x). Standard normal on the state manifold.

        `generator` is a CPU generator (that is what `torch.Generator()` gives by default and
        what makes a run reproducible across devices), so we sample on CPU and then move.
        Sampling directly on CUDA with a CPU generator is an error.
        """
        x = torch.randn(n, self._state_dim, generator=generator)
        return x.to(self.device)

    def to(self, device: torch.device | str) -> SyntheticOracle:
        """Move the operator's parameters. Returns self (in-place), matching nn.Module."""
        self.device = torch.device(device)
        for name in ("P", "W", "b", "w_succ", "w_margin", "w_slip"):
            setattr(self, name, getattr(self, name).to(self.device))
        return self
