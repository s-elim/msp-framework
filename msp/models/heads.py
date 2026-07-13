import torch
import torch.nn as nn
from typing import Dict

from msp.models import HEAD_REGISTRY

class BaseOutcomeHead(nn.Module):
    """Abstract interface for predicting physics outcomes."""
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@HEAD_REGISTRY.register()
class MLPOutcomeHead(BaseOutcomeHead):
    """
    Standard MLP implementation mapping Belief (Z) and Action (A) to outcomes (Y).
    p_psi(y | z, a)
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 7):
        super().__init__()
        
        # Fusing z and a
        input_dim = latent_dim + action_dim
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Branches for different outcome parameters
        self.succ_branch = nn.Linear(128, 1) # Logit for success
        
        # Margin branch (predicts mean and log-variance for Gaussian)
        self.margin_branch = nn.Linear(128, 2)
        
        # Slip branch (predicts mean and log-variance for Gaussian)
        self.slip_branch = nn.Linear(128, 2)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Latent statistic. Shape: (B, d) or (B, K, d)
            a (torch.Tensor): Action vector. Shape: (B, N_a, action_dim)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing outcome distribution parameters.
        """
        # Broadcasting logic to handle multiple actions evaluated against the same belief
        # z: (B, d) -> (B, 1, d) -> (B, N_a, d)
        B, N_a, act_dim = a.shape
        if z.dim() == 2:
            z_expanded = z.unsqueeze(1).expand(-1, N_a, -1)
        else:
            # If Z already has a sample dimension (e.g., K posterior samples)
            z_expanded = z
            
        # Concat along the feature dimension
        fused_input = torch.cat([z_expanded, a], dim=-1) # (B, N_a, d + act_dim)
        
        # Flatten for MLP processing
        flat_input = fused_input.view(-1, fused_input.shape[-1])
        
        shared_feat = self.shared_mlp(flat_input)
        
        succ_logit = self.succ_branch(shared_feat).view(*fused_input.shape[:-1], 1)
        margin_out = self.margin_branch(shared_feat).view(*fused_input.shape[:-1], 2)
        slip_out = self.slip_branch(shared_feat).view(*fused_input.shape[:-1], 2)
        
        return {
            "succ_logit": succ_logit,
            "margin_mu": margin_out[..., 0:1],
            "margin_logvar": margin_out[..., 1:2],
            "slip_mu": slip_out[..., 0:1],
            "slip_logvar": slip_out[..., 1:2],
        }


class AcquisitionNet(nn.Module):
    """
    Amortized estimator for Value of Information (alpha_omega).
    Predicts expected Information Gain (IG) for a viewpoint action b.
    """
    def __init__(self, obs_feature_dim: int = 512, viewpoint_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_feature_dim + viewpoint_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Predicts IG score
        )
        
    def forward(self, obs_features: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_features: (B, obs_feature_dim)
            b: Camera viewpoints (B, N_b, viewpoint_dim)
        """
        B, N_b, v_dim = b.shape
        obs_expanded = obs_features.unsqueeze(1).expand(-1, N_b, -1)
        fused = torch.cat([obs_expanded, b], dim=-1)
        return self.net(fused)
