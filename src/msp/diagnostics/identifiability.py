"""The outcome Jacobian, its null space, and the subspace-alignment test.

Formalization Definition 9.1, Eq 25, Theorem 4, and the Corollary.

    J(x) : T_x X -> (+)_j R^p,     J(x) delta = ( D_x theta_j(x)[delta] )_j
    T_x [x] = ker J(x),            dim [x] = d_X - rank J(x)

This module implements contribution C2, which had no implementation at all in the audited
repository. It produces the paper's decisive identifiability evidence: the principal angles
between the *predicted* null space ker J(x) and the *measured* grasp-invariant perturbation
subspace. Tight alignment is what turns Theorem 4 from a correct statement into a
demonstrated property.

Everything here is validated against `SyntheticOracle`, whose null space is known in closed
form, before it is pointed at a simulator. That is the only way to distinguish "the theory
is wrong" from "my estimator is broken."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.oracle.base import PhysicsOracle

__all__ = [
    "IdentifiabilityReport",
    "outcome_jacobian",
    "null_space",
    "row_space",
    "principal_angles",
    "subspace_alignment",
    "measure_invariant_subspace",
    "analyze",
]


@dataclass(frozen=True)
class IdentifiabilityReport:
    """The numbers the paper reports for Theorem 4."""

    state_dim: int
    numerical_rank: int
    null_dim: int
    singular_values: Tensor
    #: Principal angles (radians) between predicted ker J(x) and the measured
    #: grasp-invariant subspace. All near zero => Theorem 4 is demonstrated.
    principal_angles_rad: Tensor | None = None

    @property
    def max_angle_deg(self) -> float | None:
        if self.principal_angles_rad is None:
            return None
        return float(torch.rad2deg(self.principal_angles_rad.max()))

    def to_metrics(self) -> dict[str, float]:
        m = {
            "identifiability/rank": float(self.numerical_rank),
            "identifiability/null_dim": float(self.null_dim),
        }
        if self.principal_angles_rad is not None:
            m["identifiability/max_principal_angle_deg"] = self.max_angle_deg or 0.0
            m["identifiability/mean_principal_angle_deg"] = float(
                torch.rad2deg(self.principal_angles_rad).mean()
            )
        return m


def outcome_jacobian(
    oracle: PhysicsOracle,
    state: Tensor,
    actions: Tensor,
    *,
    eps: float = 5e-3,
) -> Tensor:
    """J(x), the differential of the outcome map Phi. Formalization Eq 25.

    Uses autodiff when the oracle supports it (exact), and central finite differences
    otherwise (a black-box simulator). Section 11 prescribes exactly this: "estimate J(x)
    by autodiff of Phi in sim".

    Args:
        oracle: provides Phi via `outcome_params`.
        state:  (state_dim,) a SINGLE state. J is a property of a point on the manifold.
        actions: (1, Na, action_dim) -- the rho-dense action set of Definition 9.1. The more
            actions, the better J is resolved; too few and the null space looks artificially
            large because the action set simply failed to probe some directions.
        eps: central-difference step, used only on the non-differentiable path.

            NUMERICS. The total error of a central difference is
            O(eps^2 * |Phi'''|) from truncation plus O(eps_machine / eps) from roundoff,
            minimized at eps ~ eps_machine^(1/3). In float32 (eps_machine ~ 1.2e-7) that is
            ~5e-3, which is the default. A "small" step like 1e-4 is far BELOW the optimum
            and measures floating-point noise, not the derivative -- it produced ~1e-2
            absolute error here. Prefer the autodiff path whenever the oracle supports it.

    Returns:
        (Na * P, state_dim) -- the Jacobian.
    """
    if state.dim() != 1:
        raise ValueError(f"state must be a single point (state_dim,); got {tuple(state.shape)}")

    def phi(x: Tensor) -> Tensor:
        return oracle.outcome_params(x.unsqueeze(0), actions).squeeze(0)

    if oracle.differentiable:
        return torch.autograd.functional.jacobian(phi, state, vectorize=True)  # type: ignore[no-any-return]

    # Central differences, batched: one oracle call for all 2*d perturbed states.
    d = state.numel()
    basis = torch.eye(d, device=state.device, dtype=state.dtype) * eps
    plus = state.unsqueeze(0) + basis  # (d, state_dim)
    minus = state.unsqueeze(0) - basis
    both = torch.cat([plus, minus], dim=0)  # (2d, state_dim)
    acts = actions.expand(2 * d, -1, -1)
    out = oracle.outcome_params(both, acts)  # (2d, out_dim)
    return ((out[:d] - out[d:]) / (2 * eps)).T.contiguous()  # (out_dim, state_dim)


def null_space(J: Tensor, rtol: float = 1e-5) -> Tensor:
    """ker J(x): the manipulation-INDISTINGUISHABLE directions. Theorem 4.

    Perturbing x along these directions changes no outcome of any action. Any pose or shape
    error measured along them is, by Theorem 4's Corollary, *unidentifiable and
    task-irrelevant* -- which is the formal reason ADD-S and Chamfer misrank methods.

    Returns:
        (state_dim, null_dim) orthonormal basis. Empty second axis if J has full rank.
    """
    _, s, vh = torch.linalg.svd(J, full_matrices=True)
    tol = rtol * s.max() if s.numel() > 0 else torch.tensor(0.0, device=J.device)
    rank = int((s > tol).sum().item())
    return vh[rank:].T.contiguous()


def row_space(J: Tensor, rtol: float = 1e-5) -> Tensor:
    """row J(x): the IDENTIFIABLE directions. (state_dim, rank).

    The only component of a pose/shape readout that is meaningful to report.
    """
    _, s, vh = torch.linalg.svd(J, full_matrices=True)
    tol = rtol * s.max() if s.numel() > 0 else torch.tensor(0.0, device=J.device)
    rank = int((s > tol).sum().item())
    return vh[:rank].T.contiguous()


def principal_angles(A: Tensor, B: Tensor) -> Tensor:
    """Principal angles between two subspaces, given orthonormal bases A and B.

    theta_k = arccos(sigma_k) where sigma_k are the singular values of A^T B.
    All angles ~ 0 means the subspaces coincide. This is the quantity the T-RO review
    identifies as the decisive identifiability evidence.

    Args:
        A: (n, p) orthonormal columns.
        B: (n, q) orthonormal columns.

    Returns:
        (min(p, q),) angles in radians, ascending.
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"ambient dims differ: {A.shape[0]} vs {B.shape[0]}")
    if A.numel() == 0 or B.numel() == 0:
        return torch.empty(0, device=A.device)
    s = torch.linalg.svdvals(A.T @ B).clamp(-1.0, 1.0)
    return torch.arccos(s)


def subspace_alignment(A: Tensor, B: Tensor) -> float:
    """A scalar summary in [0, 1]: 1.0 iff the subspaces coincide.

    mean(cos^2(theta_k)) -- the normalized projection energy. Reported alongside the angles
    because a single number is what goes in an abstract.
    """
    ang = principal_angles(A, B)
    if ang.numel() == 0:
        return 0.0
    return float((torch.cos(ang) ** 2).mean())


@torch.no_grad()
def measure_invariant_subspace(
    oracle: PhysicsOracle,
    state: Tensor,
    actions: Tensor,
    *,
    n_probes: int = 512,
    delta: float = 5e-3,
    rtol: float = 1e-3,
) -> Tensor:
    """The EMPIRICALLY grasp-invariant perturbation subspace, measured WITHOUT gradients.

    This is the real-robot perturbation-invariance probe of V4, and it must be genuinely
    independent of `outcome_jacobian` -- otherwise the subspace-alignment experiment is a
    tautology (comparing the Jacobian's null space to itself) rather than evidence.

    HOW NOT TO DO IT. The obvious approach -- sample random directions, keep the ones whose
    outcome response is below a threshold -- does not work, and the failure is instructive.
    ker J(x) is a measure-zero subspace of R^d, so a random direction lands in it with
    probability zero. Rejection sampling finds nothing, no matter how many probes you draw.

    WHAT WORKS. The squared outcome response to a unit perturbation u is a quadratic form::

        r(u)^2  =  || Phi(x + delta*u) - Phi(x - delta*u) ||^2 / (2*delta)^2
                ~= || J u ||^2
                 = u^T (J^T J) u   =:  u^T G u

    G is a symmetric d x d matrix with d(d+1)/2 unknowns, and every probe gives one linear
    measurement of it. With n_probes >> d(d+1)/2 we recover G by least squares, then read the
    invariant subspace off its small eigenvalues. G shares its null space with J, so this
    recovers ker J(x) exactly -- from black-box outcome queries alone, with no derivative and
    no knowledge of the oracle's internals.

    Args:
        n_probes: must exceed d(d+1)/2 for the fit to be determined.
        delta: symmetric probe magnitude. Same float32 optimality argument as
            `outcome_jacobian`; do not shrink it "for accuracy".
        rtol: eigenvalues below rtol * lambda_max are treated as zero.

    Returns:
        (state_dim, k) orthonormal basis of the measured invariant subspace.
    """
    d = state.numel()
    n_unknowns = d * (d + 1) // 2
    if n_probes < 2 * n_unknowns:
        raise ValueError(
            f"n_probes={n_probes} is too few to identify the {n_unknowns}-parameter response "
            f"form for state_dim={d}. Use at least {2 * n_unknowns}."
        )

    u = torch.randn(n_probes, d, device=state.device, dtype=state.dtype)
    u = u / u.norm(dim=1, keepdim=True)

    # Batched central probe: 2 * n_probes oracle queries in two calls.
    plus = state.unsqueeze(0) + delta * u
    minus = state.unsqueeze(0) - delta * u
    acts = actions.expand(n_probes, -1, -1)
    resp = (oracle.outcome_params(plus, acts) - oracle.outcome_params(minus, acts)) / (
        2.0 * delta
    )
    y = (resp**2).sum(dim=-1)  # (n_probes,)  ~= u^T G u

    # Design matrix for the symmetric quadratic form:
    #   u^T G u = sum_j G_jj u_j^2 + sum_{j<k} 2 G_jk u_j u_k
    iu, ju = torch.triu_indices(d, d, device=state.device)
    coeff = torch.where(iu == ju, 1.0, 2.0).to(state.dtype)  # off-diagonals appear twice
    design = u[:, iu] * u[:, ju] * coeff  # (n_probes, n_unknowns)

    g_vec = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution.squeeze(-1)

    G = torch.zeros(d, d, device=state.device, dtype=state.dtype)
    G[iu, ju] = g_vec
    G = G + G.T - torch.diag(torch.diag(G))  # symmetrize

    evals, evecs = torch.linalg.eigh(G)  # ascending
    lam_max = evals.abs().max().clamp_min(1e-12)
    k = int((evals.abs() < rtol * lam_max).sum().item())
    return evecs[:, :k].contiguous()


def analyze(
    oracle: PhysicsOracle,
    state: Tensor,
    actions: Tensor,
    *,
    measure_empirical: bool = True,
    rtol: float = 1e-5,
) -> IdentifiabilityReport:
    """Full Theorem 4 diagnostic at one state. This is the paper's headline C2 experiment."""
    J = outcome_jacobian(oracle, state, actions)
    s = torch.linalg.svdvals(J)
    tol = rtol * s.max()
    rank = int((s > tol).sum().item())
    N = null_space(J, rtol=rtol)

    angles = None
    if measure_empirical and N.shape[1] > 0:
        measured = measure_invariant_subspace(oracle, state, actions)
        if measured.shape[1] > 0:
            angles = principal_angles(N, measured)

    return IdentifiabilityReport(
        state_dim=state.numel(),
        numerical_rank=rank,
        null_dim=N.shape[1],
        singular_values=s,
        principal_angles_rad=angles,
    )
