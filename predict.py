import os
from SimilarityModel import ViT_Sim
from DataLoader import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch

criterion = nn.CosineEmbeddingLoss()
batch_size = 8
model = models.maxvit_t('IMAGENET1K_V1').to('cuda')
#model = ViT_Sim('IMAGENET1K_SWAG_E2E_V1').to('cuda')
model.load_state_dict(torch.load('weights2/weights_1-14_9.pt'))
cos = nn.CosineSimilarity(dim=1)

for img_1, img_2, label in DataLoader('./eval', './target', 7, batch_size):
    with torch.no_grad():
        model.eval()
        pred1, pred2 = model(img_1), model(img_2)
        loss = criterion(pred1, pred2, label)
        #print(cos(pred1, pred2))
        #print(label)
        print('loss: {}'.format(loss))
                
