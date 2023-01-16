import torchvision.models as models
import torch.nn as nn

class Sim_Model(nn.Module):
    def __init__(self, weight=None):
        super(Sim_Model, self).__init__()
        self.base_encoder = models.maxvit_t(weights=weight)
        self.projection_layer = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=True),
        )

    def forward(self, x, y):
        feat_x = self.base_encoder(x)
        feat_y = self.base_encoder(y)
        out_x = self.projection_layer(feat_x)
        out_y = self.projection_layer(feat_y)
        return out_x, out_y
        


