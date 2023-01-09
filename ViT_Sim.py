import torchvision.models as models
import torch
import torch.nn.functional as F
import os, cv2
import torch.nn as nn
import numpy as np

class ViT_Sim(nn.Module):
    def __init__(self, weight):
        super(ViT_Sim, self).__init__()
        self.model = models.vit_l_16(weights=weight)
        self.similarity_layers = nn.Sequential(
            nn.Linear(1000, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 1, bias=True),
            
        )

    def feature_difference(x, y):
        return torch.abs(x - y)

    def forward(self, x, y):
        feat_x = self.model(x)
        feat_y = self.model(y)
        feat_diff = self.feature_difference(feat_x, feat_y)
        return self.similarity_layers(feat_diff)
        


