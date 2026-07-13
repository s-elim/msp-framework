import torch
import torch.nn as nn
from typing import Tuple

from msp.models import ENCODER_REGISTRY
from msp.models.backbones.base import BaseBackbone

class BaseEncoder(nn.Module):
    """
    Abstract interface for Belief Encoders mapping observations to statistics.
    """
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@ENCODER_REGISTRY.register()
class GaussianBeliefEncoder(BaseEncoder):
    """
    Maps high-dimensional observation features to the sufficient statistic Z.
    Z ~ N(mu_theta(o), diag(sigma_theta(o)^2))
    """
    def __init__(self, backbone: BaseBackbone, latent_dim: int = 128):
        """
        Args:
            backbone (BaseBackbone): The initialized feature extractor.
            latent_dim (int): Dimensionality (d) of the sufficient statistic Z.
        """
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        
        # MLPs to regress the mean and log-variance
        feature_dim = backbone.output_dim
        self.mu_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Observation (B, C, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - mu: (B, latent_dim)
                - logvar: (B, latent_dim)
        """
        features = self.backbone(x)
        mu = self.mu_net(features)
        logvar = self.logvar_net(features)
        
        # Clamp logvar to prevent numerical instability during reparameterization
        logvar = torch.clamp(logvar, min=-20.0, max=5.0)
        
        return mu, logvar
