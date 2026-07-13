import torch
import torchvision.transforms.functional as F
from typing import Any, Callable, List

class Compose:
    """Composes multiple transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

class RandomColorJitter:
    """Applies random color jitter to the RGB channels only."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be [4, H, W] where [:3] is RGB and [3] is Depth
        rgb = x[:3]
        depth = x[3:]
        
        # Apply jitter (expects [C, H, W] where C=3)
        # Note: In a true pipeline we use randomized values per call
        factor_b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        factor_c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        
        rgb = F.adjust_brightness(rgb, factor_b)
        rgb = F.adjust_contrast(rgb, factor_c)
        
        return torch.cat([rgb, depth], dim=0)

class RandomDepthNoise:
    """Injects Gaussian noise into the depth channel to simulate Sim2Real gap."""
    def __init__(self, std_dev: float = 0.002): # 2mm noise default
        self.std_dev = std_dev

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:3]
        depth = x[3:]
        noise = torch.randn_like(depth) * self.std_dev
        depth_noisy = depth + noise
        return torch.cat([rgb, depth_noisy], dim=0)

def build_transforms(is_train: bool = True) -> Callable:
    """
    Constructs the augmentation pipeline.
    """
    if is_train:
        return Compose([
            RandomColorJitter(),
            RandomDepthNoise()
        ])
    else:
        return Compose([])  # No augmentation during eval
