import torch
import torch.nn as nn
from typing import Dict, Tuple

class VIBLoss(nn.Module):
    """
    Computes the Variational Information Bottleneck objective (Eq 10, 11).
    L = Distortion + (1/beta) * Rate
    """
    def __init__(self, beta_succ: float = 1.0, beta_margin: float = 1.0, beta_slip: float = 1.0):
        super().__init__()
        self.beta = {
            "success": beta_succ,
            "margin": beta_margin,
            "slip": beta_slip
        }
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def _compute_distortion(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Measures how well the sufficient statistic predicts the physical outcomes.
        """
        # Binary Cross Entropy for grasp success
        loss_succ = self.bce(y_pred["succ_logit"], y_true["success"])
        
        # Gaussian Negative Log-Likelihood for continuous variables (margin, slip)
        def gauss_nll(y, mu, logvar):
            return 0.5 * (torch.exp(-logvar) * (y - mu)**2 + logvar)
            
        loss_margin = gauss_nll(y_true["margin"], y_pred["margin_mu"], y_pred["margin_logvar"])
        loss_slip = gauss_nll(y_true["slip"], y_pred["slip_mu"], y_pred["slip_logvar"])
        
        # Average over batch and actions
        return {
            "success": loss_succ.mean(),
            "margin": loss_margin.mean(),
            "slip": loss_slip.mean()
        }

    def _compute_rate(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes KL(q(z|o) || r(z)) where r(z) = N(0, I).
        Minimizing this restricts the information capacity of Z (minimality).
        """
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=-1)
        return kl.mean()

    def forward(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor], 
                mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            y_pred: Predictions from the OutcomeHead.
            y_true: Ground truth outcomes from the Physics Oracle.
            mu: Mean of the belief Z.
            logvar: Log-variance of the belief Z.
            
        Returns:
            loss: The scalar loss to backpropagate.
            metrics: Dict of individual loss components for logging.
        """
        dist_dict = self._compute_distortion(y_pred, y_true)
        rate = self._compute_rate(mu, logvar)
        
        # Weighted distortion sum
        total_dist = (
            dist_dict["success"] / self.beta["success"] + 
            dist_dict["margin"] / self.beta["margin"] + 
            dist_dict["slip"] / self.beta["slip"]
        )
        
        total_loss = total_dist + rate
        
        metrics = {
            "loss/total": total_loss.item(),
            "loss/distortion_succ": dist_dict["success"].item(),
            "loss/distortion_margin": dist_dict["margin"].item(),
            "loss/distortion_slip": dist_dict["slip"].item(),
            "loss/rate_kl": rate.item()
        }
        
        return total_loss, metrics
