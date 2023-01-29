import torchvision.models as models
import torch.nn as nn

class Sim_Model(nn.Module):
    def __init__(self, weight=None):
        super(Sim_Model, self).__init__()
        self.base_encoder = models.vit_l_16()
        self.projection_layer = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=True)
        )

    def forward(self, x):
        feat = self.base_encoder(x)
        out = self.projection_layer(feat)
        return out
        


