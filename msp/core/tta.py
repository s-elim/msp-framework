import torch
import torch.optim as optim
from typing import Dict, Tuple

class TTAOptimizer:
    """
    Test-Time Adaptation. Updates the belief distribution (mu, logvar) using
    a physical probe's outcome without changing the global network weights.
    (Eq 20).
    """
    def __init__(self, head: torch.nn.Module, lr: float = 1e-2, trust_region: float = 0.1):
        self.head = head
        self.lr = lr
        self.trust_region = trust_region
        self.bce = torch.nn.BCEWithLogitsLoss()

    def optimize_belief(self, mu_0: torch.Tensor, logvar_0: torch.Tensor, 
                        action_p: torch.Tensor, outcome_p: float, 
                        steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the belief to incorporate the physical outcome.
        
        Args:
            mu_0: Initial belief mean (B, d)
            logvar_0: Initial belief logvar (B, d)
            action_p: The executed probe action (B, 1, act_dim)
            outcome_p: The physical success result (1.0 or 0.0)
            
        Returns:
            Adapted mu and logvar.
        """
        # Detach from computation graph and set requires_grad to optimize them directly
        mu_t = mu_0.detach().clone().requires_grad_(True)
        logvar_t = logvar_0.detach().clone().requires_grad_(True)
        
        optimizer = optim.Adam([mu_t, logvar_t], lr=self.lr)
        
        target = torch.full((mu_0.size(0), 1, 1), outcome_p, device=mu_0.device)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Forward pass through the head (no reparameterization here, just mean for fast TTA)
            preds = self.head(mu_t.unsqueeze(1), action_p)
            
            # Outcome log-likelihood
            loss_nll = self.bce(preds["succ_logit"], target)
            
            # Trust region: KL divergence from the original belief
            # KL( N(mu_t, var_t) || N(mu_0, var_0) )
            var_t = torch.exp(logvar_t)
            var_0 = torch.exp(logvar_0)
            kl = 0.5 * torch.sum(
                logvar_0 - logvar_t + (var_t + (mu_t - mu_0)**2) / var_0 - 1, dim=-1
            ).mean()
            
            loss = loss_nll + self.trust_region * kl
            loss.backward()
            optimizer.step()
            
        return mu_t.detach(), logvar_t.detach()
