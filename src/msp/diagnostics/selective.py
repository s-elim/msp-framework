"""Selective prediction: the deployment question, asked without a degenerate answer.

WHY THIS MODULE EXISTS.

The conformal certificate of Eq 24 fires only when the prediction set is the singleton {1}:

    1 in C(o,a)   <=>  1 - s <= q_hat  <=>  s >= 1 - q_hat
    0 not in C    <=>      s >  q_hat
    ------------------------------------------------------
    certify       <=>      s >  max(q_hat, 1 - q_hat)  >=  0.5     for ANY q_hat

So no grasp is EVER certified unless the belief thinks it is more likely than not to succeed. That
is the correct meaning of "certified success" -- it is not a bug -- but on this corpus it produces a
number that is easy to mistake for a passing experiment::

    coverage 0.9056  (target 0.90)   <- the theorem holds
    abstention 1.000                 <- and the system never acts

A system that never acts is never wrong. Coverage alone cannot distinguish a calibrated certificate
from a catatonic one, and a referee will say so in one line.

WHY IT ABSTAINS, MEASURED -- AND THE TEMPTING WRONG ANSWER.

Over 16000 test actions, s tops out at 0.5005 and the certification threshold is
max(q_hat, 1-q_hat) = 0.5075. Zero actions clear it. The obvious suspect is the VIB: with beta = 30
the posterior over z is deliberately noisy, and averaging sigmoid over a noisy logit pulls
E[sigmoid] toward 0.5, so one naturally concludes that the belief's own regulariser is crushing its
confidence. THAT IS WRONG, and it is worth recording because it is the kind of story that survives
into a paper unchecked. Removing the latent noise entirely does not move the ceiling::

    max E_z[sigmoid(logit)]   (what Eq 13 uses)   0.4881
    max sigmoid(E_z[logit])   (noise removed)     0.4878

The mean logit never goes positive either. The real reason is simply that s is TELLING THE TRUTH.
Its reliability diagram is close to the diagonal, and among the top 50 grasps it ranks, it predicts
0.460 and the simulator delivers 0.520. Under a proposal distribution deliberately widened to give
the corpus failures (Assumptions A3/A8 -- roughly a 10% base rate), the best grasp available really
is close to a coin flip. Eq 24 then abstains BECAUSE THE MODEL IS HONEST, not because it is broken.

Two consequences follow, and neither is "raise alpha":

  * The certification window (max(q_hat, 1-q_hat), 1] is WIDEST at q_hat = 0.5 and narrows in both
    directions, so a WEAKER guarantee makes certification HARDER. That is why alpha = 0.05, 0.10 and
    0.20 all abstain identically at 1.000.
  * A non-vacuous certificate needs candidate grasps that are genuinely good, i.e. a tighter
    DEPLOYMENT-time proposal. The training corpus must stay wide; the candidate set at deployment
    need not, and conflating the two is the actual error.

WHAT TO REPORT INSTEAD.

The deployment question is not "is this one grasp certified?" but:

    Given a scene and a set of candidate grasps, does the system pick one, and does the object
    come off the table when it does?

That is standard selective prediction, and it has two axes, not one:

    act rate  -- the fraction of scenes the system is willing to act on
    precision -- among the scenes it acted on, the fraction where the chosen grasp SUCCEEDED

Sweeping the threshold traces a risk-coverage curve, and two scorers are compared at MATCHED act
rate. This is immune to the degeneracy above: a scorer that abstains everywhere simply has no curve.
It also asks the analytic proxy the question it actually loses on -- not "is your epsilon
calibrated?" but "when you commit to your favourite grasp, does it work?"

The single most interpretable point on the curve is act_rate = 1: ALWAYS take the scorer's top-ranked
executable grasp. That number needs no conformal machinery, no threshold, and no calibration, and it
is the one a roboticist cares about.

WHAT THIS MEASURES ON THE LIBERO GROCERIES (1998 scenes)::

    ALWAYS ACT                                  SELECTIVE
      random pick               0.122             act rate   analytic     MSP
      Ferrari-Canny on the OBB  0.161               0.10       0.235     0.701
      MSP                       0.256               0.25       0.249     0.631
                                                    0.50       0.192     0.466
                                                    1.00       0.161     0.256

The analytic proxy's precision is FLAT in the act rate (0.235 -> 0.161): becoming more selective buys
it nothing, because its confidence carries no information about whether the grasp will hold. It is
also barely above the random-pick control, which is why that control is here. MSP's precision rises
monotonically, 0.256 -> 0.701. That is a working selective classifier standing next to one that is
not, and it is the deployment-facing form of the L1 result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = ["RiskCoverage", "SelectiveComparison", "risk_coverage", "compare_selective"]

#: Sentinel used to remove non-executable actions from the argmax. A grasp the gripper cannot
#: reach is not a candidate, and a deployed system knows its own kinematics, so it is entitled to
#: filter these BEFORE choosing. Leaving them in would score a collision checker, not a grasp scorer.
_UNREACHABLE = -float("inf")


@dataclass(frozen=True)
class RiskCoverage:
    """One scorer's risk-coverage curve, plus the point that matters most."""

    act_rate: np.ndarray  # fraction of scenes acted on, ascending
    precision: np.ndarray  # success rate among acted scenes, at that act rate
    top1_success: float  # precision at act_rate = 1: always take the best grasp
    n_scenes: int

    def precision_at(self, target_act_rate: float) -> float:
        """Precision at (the closest achievable) act rate. Used to compare two scorers fairly."""
        if len(self.act_rate) == 0:
            return float("nan")
        i = int(np.argmin(np.abs(self.act_rate - target_act_rate)))
        return float(self.precision[i])


@dataclass(frozen=True)
class SelectiveComparison:
    analytic: RiskCoverage
    msp: RiskCoverage
    random_pick: float  # success of picking an executable grasp uniformly at random
    n_scenes: int

    def summary(self) -> str:
        lines = [
            f"{self.n_scenes} scenes with at least one executable grasp",
            "",
            "  ALWAYS ACT -- take the scorer's top-ranked executable grasp, every scene:",
            f"      random pick                  {self.random_pick:.3f}",
            f"      Ferrari-Canny on the OBB     {self.analytic.top1_success:.3f}",
            f"      MSP                          {self.msp.top1_success:.3f}",
            "",
            "  SELECTIVE -- precision among the scenes the system chose to act on:",
            "      act rate     analytic     MSP",
        ]
        for r in (0.1, 0.25, 0.5, 0.75, 1.0):
            lines.append(
                f"        {r:4.2f}         {self.analytic.precision_at(r):.3f}      "
                f"{self.msp.precision_at(r):.3f}"
            )
        lines += [
            "",
            "  A scorer that cannot beat 'random pick' at act_rate = 1 is not selecting grasps;",
            "  it is choosing arbitrarily among them.",
        ]
        return "\n".join(lines)

    def to_metrics(self) -> dict[str, float]:
        return {
            "selective/analytic_top1": self.analytic.top1_success,
            "selective/msp_top1": self.msp.top1_success,
            "selective/random_top1": self.random_pick,
            "selective/analytic_prec_at_25": self.analytic.precision_at(0.25),
            "selective/msp_prec_at_25": self.msp.precision_at(0.25),
            "selective/n_scenes": float(self.n_scenes),
        }


@torch.no_grad()
def risk_coverage(score: Tensor, succ: Tensor, executable: Tensor, n_points: int = 50) -> RiskCoverage:
    """Trace precision against act rate for one scorer.

    The system sees a scene, ranks its candidate grasps, and acts on the best one -- but only if
    that best one clears a confidence threshold. Sweeping the threshold sweeps the act rate.

    Args:
        score:      (B, Na) the scorer's value for each candidate grasp.
        succ:       (B, Na) the simulator's verdict, 0/1.
        executable: (B, Na) whether the gripper can reach the grasp at all.
        n_points:   how many thresholds to trace.

    Returns:
        RiskCoverage. Scenes with no executable grasp are dropped: there is no decision to make.
    """
    ex = executable.bool()
    keep = ex.any(dim=1)  # scenes where SOME grasp is reachable
    if not bool(keep.any()):
        return RiskCoverage(np.array([]), np.array([]), float("nan"), 0)

    s = score[keep].float().clone()
    y = succ[keep].float()
    e = ex[keep]
    s[~e] = _UNREACHABLE  # an unreachable grasp is not a candidate

    best = s.argmax(dim=1)  # the grasp the system would execute
    conf = s.gather(1, best[:, None]).squeeze(1)  # its confidence in that choice
    won = y.gather(1, best[:, None]).squeeze(1)  # did it actually work?

    n = int(keep.sum())
    # Thresholds spanning the confidence range. Sorting the confidences and cutting at each rank
    # gives exactly the achievable act rates, with no interpolation artefacts.
    order = conf.argsort(descending=True)
    won_sorted = won[order].cpu().numpy()

    ks = np.unique(np.linspace(1, n, min(n_points, n)).astype(int))
    act = ks / n
    prec = np.array([won_sorted[:k].mean() for k in ks])

    return RiskCoverage(
        act_rate=act,
        precision=prec,
        top1_success=float(won.mean()),
        n_scenes=n,
    )


@torch.no_grad()
def compare_selective(
    analytic_score: Tensor,
    msp_score: Tensor,
    succ: Tensor,
    executable: Tensor,
    generator: torch.Generator | None = None,
) -> SelectiveComparison:
    """Compare the analytic proxy and MSP as GRASP SELECTORS, which is how they are deployed.

    The random-pick control is not decoration. Success is rare, so a scorer can look respectable on
    a threshold metric while ranking no better than chance; comparing against a uniform pick among
    the executable grasps is what exposes that.
    """
    ex = executable.bool()
    keep = ex.any(dim=1)

    # Random control: choose uniformly among the EXECUTABLE grasps of each scene.
    e = ex[keep].float()
    y = succ[keep].float()
    idx = torch.multinomial(e, num_samples=1, generator=generator).squeeze(1)
    rand = float(y.gather(1, idx[:, None]).squeeze(1).mean())

    return SelectiveComparison(
        analytic=risk_coverage(analytic_score, succ, executable),
        msp=risk_coverage(msp_score, succ, executable),
        random_pick=rand,
        n_scenes=int(keep.sum()),
    )
