import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DataLoader import DataLoader
from SimilarityModel import SimilarityModel
import torchvision.models as models

def train(model, criterion, optimizer, batch_size, epoch, nf):
    model.train()
    for e in range(epoch):
        for img_1, img_2, label in DataLoader('./source', './target', nf, batch_size):
            optimizer.zero_grad()
            pred1, pred2 = model(img_1), model(img_2)
            loss = criterion(pred1, pred2, label)
            loss.backward()
            optimizer.step()
            print('Total loss for this batch: {}'.format(loss.item()))
        save_weights(model, e)

def save_weights(model, id):
    torch.save(model.state_dict(), 'weights/weights_1-15_{}.pt'.format(id+20))

def custom_loss_func(out_1, out_2, target):
    temp = 100
    return torch.log(torch.div(torch.exp(F.cosine_similarity(out_1, out_2)/temp), torch.sum(torch.exp(F.cosine_similarity(out_1, out_2)/temp))))

if __name__ == '__main__':
    bs = 8
    epoch = 20
    learning_rate = 1e-4
    num_false = 3
    model = models.maxvit_t('IMAGENET1K_V1').to('cuda') 
    model.load_state_dict(torch.load('weights/weights_1-14_19.pt'))
    optimize = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
    loss_func = nn.CosineEmbeddingLoss(0.1)
    train(model, loss_func, optimize, bs, epoch, num_false)
    