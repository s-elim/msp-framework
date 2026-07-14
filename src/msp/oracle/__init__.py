"""The physics kernel M -- the object that DEFINES the estimand.

"Manipulation-sufficient" means sufficient *for M*. "Manipulation-indistinguishable" means M
returns the same outcome distribution. The outcome Jacobian of Theorem 4 is the Jacobian *of M*.
If M is wrong, every downstream number measures the wrong thing -- which is why the audited
repository, whose oracle returned `torch.randint`, could not have supported a single claim in the
manuscript regardless of how correct the rest of the code was.

Two oracles ship today, and they are for different jobs.

`SyntheticOracle`
    A world whose indistinguishability class is known in CLOSED FORM. Not physics. Its purpose is
    to validate the diagnostics: on a real simulator you cannot tell "the theory is wrong" apart
    from "my estimator is broken", because there is no ground truth to check against. Here there
    is, so `diagnostics.identifiability` is proven correct before it is ever pointed at physics.

`AnalyticGraspOracle`
    Real grasp mechanics: Ferrari-Canny epsilon-quality over a linearized friction cone with
    soft-finger contacts, fully differentiable in the state. This is tier 1 of the three-tier
    operator Section 11 prescribes. Tiers 2 and 3 (a version-pinned rigid-body simulator, and a
    residual fit to real grasp outcomes) are NOT built yet. Until they are, no number produced by
    this framework transfers to a real robot, and the sim-to-real sufficiency gap -- which the
    T-RO review calls the paper's decisive experiment -- cannot be measured.
"""

from msp.oracle.analytic import STATE_DIM, STATE_SLICES, AnalyticGraspOracle
from msp.oracle.libero_assets import LiberoObject, LiberoObjectLibrary
from msp.oracle.libero_sim import LiberoGraspOracle
from msp.oracle.base import PhysicsOracle
from msp.oracle.synthetic import SyntheticOracle

__all__ = [
    "STATE_DIM",
    "STATE_SLICES",
    "AnalyticGraspOracle",
    "LiberoGraspOracle",
    "LiberoObject",
    "LiberoObjectLibrary",
    "PhysicsOracle",
    "SyntheticOracle",
]
