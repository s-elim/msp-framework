"""Offline evaluation: the numbers that go in the paper.

THE COVERAGE ESTIMATOR. Theorem 7 guarantees MARGINAL coverage::

    P( succ in C(o, a) )  >=  1 - alpha,     over ALL test points.

The audited evaluator instead computed, among the actions that landed in the certified set,
the fraction whose true label was 1 -- i.e. P(succ = 1 | a in A_cert), the PRECISION of the
certified set -- and logged it against a "Target: 1 - alpha". Conditioning on selection
destroys exchangeability, so that quantity has no reason to converge to 1 - alpha at all. A
run could have reported healthy coverage while the guarantee was violated, or the reverse.

Both quantities are computed here, under their correct names, because both are interesting:
`coverage` is the theorem, `certified_precision` is the operational question ("when it says
yes, how often is it right?"). They are simply not the same number and must not be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from msp.inference.calibrator import ConformalCalibrator
from msp.inference.engine import InferenceEngine
from msp.math.bottleneck import BetaSchedule, vib_objective
from msp.types import Outcome

__all__ = ["EvalReport", "Evaluator"]


@dataclass(frozen=True)
class EvalReport:
    coverage: float  # Theorem 7: P(succ in C(o,a)). The guarantee.
    target_coverage: float  # 1 - alpha
    certified_precision: float  # P(succ=1 | certified). The operational number.
    abstention_rate: float  # fraction of scenes with an empty A_cert
    certified_fraction: float  # fraction of ACTIONS certified
    mean_ambiguity: float  # E[U(o)]
    rate: float  # R -- x-axis of the rate-distortion frontier
    distortion: float  # D -- y-axis
    n_points: int

    def to_metrics(self) -> dict[str, float]:
        return {f"eval/{k}": float(v) for k, v in self.__dict__.items()}

    def coverage_holds(self, tol: float = 0.02) -> bool:
        return self.coverage >= self.target_coverage - tol


class Evaluator:
    def __init__(self, engine: InferenceEngine, device: torch.device) -> None:
        self.engine = engine
        self.device = device

    def _batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Outcome]:
        obs = batch["observation"].to(self.device)
        actions = batch["actions"].to(self.device)
        y = Outcome(
            succ=batch["succ"].to(self.device),
            margin=batch["margin"].to(self.device),
            slip=batch["slip"].to(self.device),
        )
        return obs, actions, y

    @torch.no_grad()
    def collect_scores(self, loader: DataLoader[Any]) -> tuple[Tensor, Tensor]:
        """Flattened (s, succ) over a fold. Feeds calibration and coverage alike."""
        self.engine.encoder.eval()
        self.engine.head.eval()
        S, Y = [], []
        for batch in loader:
            obs, actions, y = self._batch(batch)
            scored = self.engine.score(self.engine.encoder(obs), actions)
            S.append(scored.stats.s.flatten().cpu())
            Y.append(y.succ.squeeze(-1).flatten().cpu())
        return torch.cat(S), torch.cat(Y)

    def calibrate(
        self, calib_loader: DataLoader[Any], calibrator: ConformalCalibrator
    ) -> float:
        """Fit q_hat on a HELD-OUT fold. Eq 22-23."""
        s, y = self.collect_scores(calib_loader)
        return calibrator.fit(s, y)

    @torch.no_grad()
    def evaluate(
        self, test_loader: DataLoader[Any], beta: BetaSchedule | None = None
    ) -> EvalReport:
        cal = self.engine.calibrator
        if cal is None or cal.q_hat is None:
            raise RuntimeError("Evaluator needs a fitted calibrator to report coverage.")

        self.engine.encoder.eval()
        self.engine.head.eval()
        beta = beta or BetaSchedule()

        n_cov = n_cov_ok = 0
        n_cert = n_cert_ok = 0
        n_actions = 0
        abstain = scenes = 0
        amb_sum = rate_sum = dist_sum = 0.0
        n_batches = 0

        for batch in test_loader:
            obs, actions, y = self._batch(batch)
            belief = self.engine.encoder(obs)
            scored = self.engine.score(belief, actions)

            s = scored.stats.s  # (B, Na)
            succ = y.succ.squeeze(-1)  # (B, Na)

            # --- Theorem 7: MARGINAL coverage, over every test point ---
            C = cal.prediction_set(s)
            covered = torch.where(succ > 0.5, C.contains_success, C.contains_failure)
            n_cov += covered.numel()
            n_cov_ok += int(covered.sum())

            # --- the operational number: precision OF the certified set ---
            certified = C.is_certified_success()
            n_cert += int(certified.sum())
            n_cert_ok += int((certified & (succ > 0.5)).sum())
            n_actions += certified.numel()

            # --- abstention: scenes where NOTHING certifies (Eq 24) ---
            scenes += s.shape[0]
            abstain += int((~certified.any(dim=1)).sum())

            amb_sum += float(scored.ambiguity.mean())

            # --- rate-distortion frontier coordinates ---
            z = belief.rsample(1).squeeze(1)
            terms = vib_objective(
                self.engine.head(z, actions), y, belief.mu, belief.logvar, beta
            )
            rate_sum += float(terms.rate)
            dist_sum += float(terms.total_distortion)
            n_batches += 1

        nb = max(1, n_batches)
        return EvalReport(
            coverage=n_cov_ok / max(1, n_cov),
            target_coverage=1.0 - cal.alpha,
            certified_precision=n_cert_ok / max(1, n_cert),
            abstention_rate=abstain / max(1, scenes),
            certified_fraction=n_cert / max(1, n_actions),
            mean_ambiguity=amb_sum / nb,
            rate=rate_sum / nb,
            distortion=dist_sum / nb,
            n_points=n_cov,
        )
