"""Split-conformal calibration with adaptive (drift-robust) operation.

Wraps `msp.math.conformal` in the stateful object the deployment loop needs, and enforces
the one assumption the mathematics cannot check for itself: FOLD DISJOINTNESS.

Theorem 7 requires the calibration fold to be exchangeable with the test point, which in
practice means the model must not have been fit on it (Assumption A5). Nothing in the
audited code checked this -- `Evaluator.calibrate` accepted whatever loader it was handed,
so passing the training loader silently produced an invalid certificate that still printed a
healthy-looking number. We fingerprint the fold and refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
from torch import Tensor

from msp.math.conformal import (
    AdaptiveConformalState,
    conformal_quantile,
    nonconformity_scores,
    prediction_set,
)
from msp.types import PredictionSet

__all__ = ["ConformalCalibrator"]


@dataclass
class ConformalCalibrator:
    """Split conformal (Eq 22-24) plus ACI (Section 7).

    Args:
        alpha: target miscoverage. 0.1 => a 90% certificate.
        gamma: ACI step size. Set to 0 to disable adaptation (pure split conformal).
    """

    alpha: float = 0.1
    gamma: float = 0.01
    q_hat: float | None = None
    _aci: AdaptiveConformalState | None = field(default=None, repr=False)
    _cal_fingerprint: int | None = field(default=None, repr=False)
    _n_cal: int = 0

    def __post_init__(self) -> None:
        if self.gamma > 0:
            self._aci = AdaptiveConformalState(target=self.alpha, gamma=self.gamma)

    # -- calibration ---------------------------------------------------------

    def fit(
        self,
        s: Tensor,
        succ: Tensor,
        *,
        train_fingerprint: int | None = None,
    ) -> float:
        """Compute q_hat on a held-out fold. Eq 22-23.

        Args:
            s:    (N,) predicted success probabilities on the CALIBRATION fold.
            succ: (N,) realized outcomes.
            train_fingerprint: hash of the training fold's sample ids. If supplied and it
                collides with this fold's, we raise: the certificate would be invalid.
        """
        s, succ = s.flatten(), succ.flatten()
        fp = self._fingerprint(s)
        if train_fingerprint is not None and fp == train_fingerprint:
            raise ValueError(
                "The calibration fold appears to be identical to the training fold. Split "
                "conformal requires them to be disjoint (Assumption A5); calibrating on "
                "training data yields a certificate that is invalid but still prints a "
                "plausible number. Pass a held-out loader."
            )

        scores = nonconformity_scores(s, succ)
        self.q_hat = conformal_quantile(scores, self.alpha)
        self._cal_fingerprint = fp
        self._n_cal = s.numel()
        return self.q_hat

    @staticmethod
    def _fingerprint(t: Tensor) -> int:
        return hash((t.numel(), float(t.sum()), float((t * t).sum())))

    # -- deployment ----------------------------------------------------------

    @property
    def effective_alpha(self) -> float:
        """The level currently in force: the ACI-adapted alpha under drift, else the target."""
        return self._aci.alpha_t if self._aci is not None else self.alpha

    def prediction_set(self, s: Tensor) -> PredictionSet:
        """C(o, a) for every candidate action. Eq 23."""
        if self.q_hat is None:
            raise RuntimeError(
                "Calibrator is not fitted. Call fit() on a held-out fold before certifying; "
                "an uncalibrated certificate is not a certificate."
            )
        return prediction_set(s, self.q_hat)

    def certified_mask(self, s: Tensor) -> Tensor:
        """A_cert: the SINGLETON test C(o,a) == {1}. Eq 24. (B, Na) bool."""
        return self.prediction_set(s).is_certified_success()

    def observe(self, s: Tensor, succ: Tensor) -> bool:
        """Feed back one realized outcome. Updates ACI and re-derives q_hat's level.

        Returns whether the point was MIScovered (the ACI err_t).
        """
        if self.q_hat is None:
            raise RuntimeError("Calibrator is not fitted.")
        C = prediction_set(s.reshape(1, -1), self.q_hat)
        covered = torch.where(
            succ.reshape(1, -1) > 0.5, C.contains_success, C.contains_failure
        )
        miscovered = not bool(covered.all())
        if self._aci is not None:
            self._aci.update(miscovered=miscovered)
        return miscovered


    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "q_hat": self.q_hat,
                    "n_cal": self._n_cal,
                    "alpha_t": self.effective_alpha,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> ConformalCalibrator:
        d = json.loads(Path(path).read_text())
        c = cls(alpha=d["alpha"], gamma=d["gamma"])
        c.q_hat = d["q_hat"]
        c._n_cal = d.get("n_cal", 0)
        if c._aci is not None:
            c._aci.alpha_t = d.get("alpha_t", d["alpha"])
        return c
