import torch
import torch.nn as nn
import torchvision.models as models


class ModelCombo(nn.Module):
    def __init__(self):
        super(ModelCombo, self).__init__()
        self.model1 = models.vit_b_16(weights='IMAGENET1K_SWAG_LINEAR_V1')
        self.model2 = models.efficientnet_v2_m(weights='DEFAULT')
        # self.model1 = models.vgg16(weights='DEFAULT')
        # self.model2 = models.resnet50(weights='IMAGENET1K_V2')
        # models.inception_v3(weights='IMAGENET1K_V1')
        self.model3 = models.swin_t(weights='DEFAULT')
        num_features = 1000
        self.fc1 = nn.Linear(num_features*3, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())

        if self.training:
            x = torch.cat((x1, x2, x3), dim=1)  # .logits
        else:
            x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
