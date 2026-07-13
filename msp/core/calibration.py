import numpy as np
import torch
import json
import os
import logging

logger = logging.getLogger(__name__)

class ConformalCalibrator:
    """
    Split conformal prediction to yield certified success bounds.
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Args:
            alpha: Target error rate (e.g. 0.1 for 90% coverage).
            gamma: Step size for Adaptive Conformal Inference (ACI).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.q_hat = None

    def fit(self, calibration_scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes the empirical quantile q_hat from a held-out calibration set.
        
        Args:
            calibration_scores: The model's predicted s(o, a).
            labels: Ground truth binary successes.
            
        Returns:
            q_hat: The computed quantile threshold.
        """
        n = len(calibration_scores)
        
        # Nonconformity score E_i (Eq 22)
        # If success (1): E_i = 1 - s_i
        # If failure (0): E_i = s_i
        E = np.where(labels == 1, 1 - calibration_scores, calibration_scores)
        
        # Quantile index
        val = np.ceil((n + 1) * (1 - self.alpha)) / n
        val = min(max(val, 0.0), 1.0) # Clamp to [0,1]
        
        self.q_hat = np.quantile(E, val)
        logger.info(f"Fitted Conformal Calibrator. alpha={self.alpha}, q_hat={self.q_hat:.4f}")
        return self.q_hat

    def get_certified_set(self, s_test: torch.Tensor) -> torch.Tensor:
        """
        Filters actions into the certified set (Eq 24).
        
        Args:
            s_test: Predicted success probabilities.
            
        Returns:
            Boolean mask where True means certified success.
        """
        if self.q_hat is None:
            raise ValueError("Calibrator has not been fitted yet.")
            
        # The success label '1' is in the set if score_1 <= q_hat
        # score_1 = 1 - s_test
        certified_mask = (1.0 - s_test) <= self.q_hat
        return certified_mask

    def update_aci(self, err_t: float) -> None:
        """
        Adaptive Conformal Inference (ACI) step.
        Updates alpha online to account for distribution drift.
        """
        self.alpha = self.alpha + self.gamma * (self.alpha - err_t)
        self.alpha = max(0.01, min(self.alpha, 0.99)) # Clamp
        
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"alpha": self.alpha, "gamma": self.gamma, "q_hat": self.q_hat}, f)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            self.alpha = data["alpha"]
            self.gamma = data["gamma"]
            self.q_hat = data["q_hat"]
