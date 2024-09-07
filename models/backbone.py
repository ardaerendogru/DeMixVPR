"""Backbone model for feature extraction using ResNet18."""

from torch import nn
from torchvision import models

class Backbone(nn.Module):
    """Backbone model using a cropped ResNet18."""
    def __init__(self, weights='ResNet18_Weights.DEFAULT', mixvpr=False):
        super().__init__()
        resnet = models.resnet18(weights=weights) 
        if mixvpr:
            self.cropped_resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            self.cropped_resnet = nn.Sequential(*list(resnet.children())[:-2])

        for i, child in enumerate(self.cropped_resnet.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass through the cropped ResNet18."""
        x = self.cropped_resnet(x)
        return x
    