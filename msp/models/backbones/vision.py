import torch
import torch.nn as nn
import torchvision.models as models

from msp.models import BACKBONE_REGISTRY
from msp.models.backbones.base import BaseBackbone

@BACKBONE_REGISTRY.register()
class ResNetBackbone(BaseBackbone):
    """
    A ResNet-based feature extractor modified to accept 4-channel input (RGB-D).
    """
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__(output_dim=output_dim)
        
        # Load standard ResNet18
        # Using weights=models.ResNet18_Weights.DEFAULT if PyTorch >= 1.13
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Modify the first convolutional layer to accept 4 channels instead of 3
        # We copy the weights from the RGB channels to the new Depth channel
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            self.conv1.weight[:, 3] = original_conv1.weight[:, 0] # Rough initialization for depth
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # Output projection to standard dimension
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps (B, 4, H, W) -> (B, output_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        
        return x
