"""The physics kernel M(y | x, a): the object that DEFINES the estimand.

Formalization Section 0::

    M : X x A -> P(Y),   M(dy | x, a)

Everything in MSP is defined relative to M. "Manipulation-sufficient" means sufficient
*for M*. "Manipulation-indistinguishable" means M gives the same outcome distribution.
The outcome Jacobian of Theorem 4 is the Jacobian *of M*. If M is wrong, every downstream
number in the paper is a measurement of the wrong thing -- which is why the audited
implementation, whose oracle returned `torch.randint`, could not have supported a single
claim in the manuscript regardless of how correct the rest of the code was.

So this interface is deliberately demanding:

* `query` returns a realized `Outcome` (a sample from M), for training supervision.
* `outcome_params` returns the *deterministic parameter vector* of M(.|x,a) -- the
  theta_j(x) of Definition 9.1 -- and must be DIFFERENTIABLE in x. This is what makes the
  identifiability diagnostic (Eq 25) possible at all: J(x) is obtained by autodiff through
  this method. An oracle that cannot provide it can still train a model but cannot support
  contribution C2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from msp.types import Outcome

__all__ = ["PhysicsOracle"]


class PhysicsOracle(ABC):
    """Ground-truth action-conditioned outcome operator."""

    #: Whether `outcome_params` is differentiable w.r.t. the state. Analytic and synthetic
    #: oracles set True; a black-box rigid-body simulator sets False and the identifiability
    #: diagnostic must then fall back to finite differences.
    differentiable: bool = False

    @abstractmethod
    def query(self, state: Tensor, actions: Tensor) -> Outcome:
        """Sample y ~ M(. | x, a).

        Args:
            state:   (B, state_dim) -- the physical state x. NOT an observation.
            actions: (B, Na, action_dim)

        Returns:
            Outcome with fields of shape (B, Na, 1). Implementations must call
            `Outcome.validate()` before returning: an oracle that emits negative slip or a
            non-binary success is a bug that must not reach the loss.
        """

    @abstractmethod
    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:
        """The deterministic parameters of M(.|x,a): theta_j(x) of Definition 9.1.

        This is Phi(x) evaluated at a finite action set -- the map whose Jacobian is the
        outcome Jacobian J(x) of Eq 25, and whose null space is the manipulation-
        indistinguishability class of Theorem 4.

        Args:
            state:   (B, state_dim), may require grad.
            actions: (B, Na, action_dim)

        Returns:
            (B, Na * P) -- the stacked outcome parameters, flattened over actions, where P
            is the number of parameters per action (success logit, margin mean, slip mean).
            Flattened because J(x) is a single linear map from the state tangent space into
            the direct sum over actions.
        """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """d_X -- the dimension of the state manifold X."""

    def null_space_dim(self) -> int | None:
        """The known dimension of ker J(x), if the oracle knows it analytically.

        Returns None for oracles that do not (simulators, real robots). `SyntheticOracle`
        overrides this, which is what lets the identifiability diagnostic be validated
        against ground truth before it is trusted on a real scene.
        """
        return None
