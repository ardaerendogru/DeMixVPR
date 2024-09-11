"""Helper module for model creation"""

from torch import nn
from .backbone import Backbone
from .aggregators import GeM, Average, MixVPR
import torch
import torch.nn.functional as F
class GetModel(nn.Module):
    """
    Model class that combines a backbone network with an aggregation layer.
    """
    def __init__(self, weights='ResNet18_Weights.DEFAULT', aggregator="average", input_size=512, output_size=512):
        self.agg_name = aggregator
        super().__init__()
        if aggregator == "mixvpr":
            self.model = Backbone(weights='ResNet18_Weights.DEFAULT', mixvpr=True)
        else:
            self.model = Backbone(weights='ResNet18_Weights.DEFAULT')
            
        if aggregator == "gem":
            self.aggregator = GeM(input_size=input_size, output_size=output_size)
            self.fc = nn.Linear(input_size, output_size)

        elif aggregator == "average":
            self.aggregator = Average(input_size=input_size, output_size=output_size)
            self.fc = nn.Linear(input_size, output_size)

        elif aggregator == "mixvpr":
            self.aggregator = MixVPR(in_channels=input_size, out_channels=output_size//4)

        else:
            raise ValueError("Aggregator can be either gem or average.")

    def forward(self, x):
        """Forward pass through the model and aggregator."""
        x = self.model(x)
        x = self.aggregator(x)
        if self.agg_name == "mixvpr":
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    

class GetMultiModel(nn.Module):
    """
    Model class that combines a backbone network with an aggregation layer.
    """
    def __init__(self, weights='ResNet18_Weights.DEFAULT', aggregator="average", input_size=1024, output_size=512, use_fusion = False, is_cos_place=False):
        super().__init__()
        if aggregator == "mixvpr":
            self.model = Backbone(weights='ResNet18_Weights.DEFAULT', mixvpr=True)
            self.model_depth = Backbone(weights='ResNet18_Weights.DEFAULT', mixvpr=True)
        else:
            self.model = Backbone(weights='ResNet18_Weights.DEFAULT')
            self.model_depth = Backbone(weights='ResNet18_Weights.DEFAULT')

        self.aggregator = aggregator
        self.is_cos_place = is_cos_place
        self.use_fusion = use_fusion
        if aggregator == "gem":
            self.aggregator_image = GeM(input_size=int(input_size/2), output_size=output_size)
            self.aggregator_depth = GeM(input_size=int(input_size/2), output_size=output_size)
            self.fc = nn.Linear(input_size, output_size)

        elif aggregator == "average":
            self.aggregator_image = Average(input_size=int(input_size/2), output_size=output_size)
            self.aggregator_depth = Average(input_size=int(input_size/2), output_size=output_size)
            self.fc = nn.Linear(input_size, output_size)

        elif aggregator == "mixvpr":
            self.aggregator_image = MixVPR(in_channels=input_size, out_channels=output_size)        
        else:
            raise ValueError("Aggregator can be either gem or average.")

            
    def forward(self, x, x_depth):
        x = self.model(x)
        x_depth = self.model_depth(x_depth)
        if self.use_fusion:
            x = self.fusion(x, x_depth)
            x = self.aggregator_image(x)
            x = self.fc(x)
            return x
        if self.aggregator == "mixvpr":
            x = torch.cat((x, x_depth), dim=1)
            x = self.aggregator_image(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        else:
            x = self.aggregator_image(x)
            x_depth = self.aggregator_depth(x_depth)
            x = torch.cat((x, x_depth), dim=1)
            x = self.fc(x)
            x = F.normalize(x, p=2, dim=1)
            return x
    

class GetDeAttentionModule(nn.Module):
    def __init__(self):
        super(GetDeAttentionModule, self).__init__()
        self.conv_rgb = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.conv_depth = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.att_conv1 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        self.att_conv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, rgb_features, depth_features):
        rgb_features = self.relu(self.conv_rgb(rgb_features))
        depth_features = self.relu(self.conv_depth(depth_features))
        
        pooled_features = rgb_features * depth_features
        att_map = self.softmax(self.att_conv1(pooled_features))
        att_map = self.sigmoid(self.att_conv2(att_map))
        
        return att_map
    


class GetDeAttNet(nn.Module):
    """
    Model class that combines a backbone network with an aggregation layer.
    """
    def __init__(self, weights='ResNet18_Weights.DEFAULT', aggregator="average", input_size=512, output_size=512):
        super().__init__()
        self.model = Backbone(weights='ResNet18_Weights.DEFAULT')
        self.model_depth = Backbone(weights='ResNet18_Weights.DEFAULT')
        if aggregator == "gem":
            self.aggregator = GeM(input_size=input_size, output_size=output_size)
        elif aggregator == "average":
            self.aggregator = Average(input_size=input_size, output_size=output_size)
        else:
            raise ValueError("Aggregator can be either gem or average.")
        self.fc = nn.Linear(input_size, output_size)
        self.GetDeAttentionModule = GetDeAttentionModule()
    def forward(self, x, x_depth):
        x = self.model(x)
        x_depth = self.model_depth(x_depth)
        attention_map = self.GetDeAttentionModule(x, x_depth)
        image_attention = x * attention_map
        x = self.aggregator(image_attention)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
