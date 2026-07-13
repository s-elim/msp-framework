import torch
import torch.nn as nn

from msp.models import READOUT_REGISTRY

class BaseReadout(nn.Module):
    """Abstract interface for optional metric decoders (e.g., pose or shape)."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@READOUT_REGISTRY.register()
class PoseReadout(BaseReadout):
    """
    Decodes the sufficient statistic Z back into an identifiable 6D pose.
    This is strictly for 'honest' evaluation tracking (Section 10 of formalization) 
    and is not used for manipulation decisions.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7) # Predicts translation (3) and quaternion (4)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Latent statistic (B, d)
        Returns:
            torch.Tensor: Evaluated pose [tx, ty, tz, qw, qx, qy, qz]
        """
        pose_raw = self.net(z)
        
        t = pose_raw[..., :3]
        q = pose_raw[..., 3:]
        
        # Normalize quaternion
        q = q / q.norm(dim=-1, keepdim=True)
        
        return torch.cat([t, q], dim=-1)
