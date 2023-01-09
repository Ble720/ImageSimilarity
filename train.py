import cv2, torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from DataLoader import DataLoader
from ViT_Sim import ViT_Sim

def train(model, criterion, optimizer, batch_size, epoch):
    for e in range(epoch):
        for img_1, img_2, label in DataLoader(batch_size):
            optimizer.zero_grad()
            pred = model(img_1, img_2)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

def save_weights(model):
    torch.save(model, 'weights.pt')

if __name__ == '__main__':
    bs = 8
    epoch = 5
    learning_rate = 1e-5
    model = ViT_Sim('IMAGENET1K_SWAG_E2E_V1')
    model.load_state_dict(torch.load('weights.pt'))
    optimize = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    train(model, loss_func, optimize, bs, epoch)
    