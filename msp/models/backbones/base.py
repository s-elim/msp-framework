import torch.nn as nn
import torch

class BaseBackbone(nn.Module):
    """
    Abstract interface for all observation feature extractors.
    
    This guarantees that any backbone (ResNet, PointNet, ViT) exposes the 
    same output signature for the downstream BeliefEncoder.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): The observation tensor. e.g., shape (B, 4, H, W)
            
        Returns:
            torch.Tensor: Feature vector of shape (B, output_dim)
        """
        raise NotImplementedError
