"""Honest evaluation of active perception. Contribution C3.

THE TRAP THIS MODULE EXISTS TO AVOID, because it is easy to fall into and it inflates the headline.

The natural way to report the value of active perception is: for each scene, compute the
information gain of all |B| candidate viewpoints, take the best, and report how much ambiguity it
removes. That number is WRONG, and wrong in the flattering direction.

U(o ∪ o_b) is a Monte Carlo estimate (K posterior samples through the outcome head), so IG(b) is
noisy. Taking a max over |B| noisy estimates selects, in part, for whichever estimate happened to be
noisiest -- the classic winner's curse. The reported gain is then partly a measurement of your own
sampling error.

Measured on the RGB-D corpus with |B| = 8, K = 32:

    random second view                                     14.6%  ambiguity reduction
    "oracle best" (select AND score on the same estimate)  43.0%
    honest best   (select on A, score on independent B)    17.5%   <-- the real number

**25.4 of the apparent 43 points were selection noise.** An earlier version that also fused views
by a product-of-Gaussians reported 74%, all of it artifact.

And read the honest row against the random row before celebrating: taking a second view AT ALL buys
14.6 points. Choosing WHICH view buys 2.9 more. That is the actual, modest, defensible claim about
active perception in this setting, and it is the one that belongs in the paper.

THE FIX is the standard one and it costs one extra forward pass: select the viewpoint using one
independent estimate, then score the selected viewpoint using a second, freshly-sampled estimate.
The noise that drove the selection is then independent of the noise in the score, and the bias is
gone by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.inference.active import compute_true_information_gain
from msp.models.nets import BeliefEncoder, OutcomeHead

__all__ = ["ActiveReport", "evaluate_active_perception"]


@dataclass(frozen=True)
class ActiveReport:
    """The numbers the paper should report for C3. All ambiguity reductions are fractions of U(o)."""

    u_one_view: float  # U(o) from a single view
    reduction_random: float  # a randomly chosen second view
    reduction_oracle_biased: float  # select and score on the SAME estimate -- DO NOT REPORT THIS
    reduction_honest: float  # select on A, score on independent B -- report THIS
    reduction_learned: float | None  # the acquisition net's own choice, scored honestly
    selection_bias: float  # oracle_biased - honest: how much of the gain was noise
    n_scenes: int
    n_views: int

    def to_metrics(self) -> dict[str, float]:
        m = {f"active/{k}": float(v) for k, v in self.__dict__.items() if v is not None}
        return m

    def summary(self) -> str:
        lines = [
            f"U(o), one view                    {self.u_one_view:.5f}",
            f"ambiguity reduction, random view  {self.reduction_random * 100:5.1f}%",
            f"ambiguity reduction, honest best  {self.reduction_honest * 100:5.1f}%   <- report this",
        ]
        if self.reduction_learned is not None:
            lines.append(
                f"ambiguity reduction, alpha_omega  {self.reduction_learned * 100:5.1f}%   "
                "<- what the acquisition net actually achieves"
            )
        lines += [
            f"('oracle best' would read {self.reduction_oracle_biased * 100:.1f}%, of which "
            f"{self.selection_bias * 100:.1f} points is max-of-N selection noise. Do not report it.)",
        ]
        return "\n".join(lines)


@torch.no_grad()
def evaluate_active_perception(
    encoder: BeliefEncoder,
    head: OutcomeHead,
    views: Tensor,
    actions: Tensor,
    acquisition: torch.nn.Module | None = None,
    num_samples: int = 32,
    seed: int = 0,
) -> ActiveReport:
    """Measure what a second viewpoint is actually worth.

    Args:
        views: (B, V, 4, H, W) -- every viewpoint of every scene. View 0 is the starting one.
        actions: (B, Na, Ad) -- the action set over which ambiguity is averaged (Eq 16's rho).
        acquisition: the fitted alpha_omega, if you want its choice scored too.
    """
    b, v = views.shape[:2]
    first = [views[:, 0]]

    # Two INDEPENDENT Monte Carlo estimates of the same quantity.
    torch.manual_seed(seed)
    ig_select, _ = compute_true_information_gain(
        encoder, head, first, views, actions, num_samples=num_samples
    )
    torch.manual_seed(seed + 10_000)
    ig_score, u0 = compute_true_information_gain(
        encoder, head, first, views, actions, num_samples=num_samples
    )

    denom = u0.clamp_min(1e-9)

    def reduction(gain: Tensor) -> float:
        return float((gain.clamp_min(0.0) / denom).mean())

    rows = torch.arange(b, device=views.device)

    rnd_idx = torch.randint(0, v, (b,), device=views.device)
    r_random = reduction(ig_score[rows, rnd_idx])

    r_biased = reduction(ig_select.max(dim=1).values)  # the number NOT to report

    honest_idx = ig_select.argmax(dim=1)
    r_honest = reduction(ig_score[rows, honest_idx])

    r_learned = None
    if acquisition is not None and bool(getattr(acquisition, "fitted", False)):
        feats = encoder.backbone(views[:, 0])
        view_ids = torch.arange(v, device=views.device).unsqueeze(0).expand(b, -1)
        pred = acquisition(feats, view_ids)  # (B, V)
        r_learned = reduction(ig_score[rows, pred.argmax(dim=1)])

    return ActiveReport(
        u_one_view=float(u0.mean()),
        reduction_random=r_random,
        reduction_oracle_biased=r_biased,
        reduction_honest=r_honest,
        reduction_learned=r_learned,
        selection_bias=r_biased - r_honest,
        n_scenes=b,
        n_views=v,
    )
