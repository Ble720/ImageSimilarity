import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DataLoader import DataLoader
from Sim_Model import Sim_Model
import torchvision.models as models

import argparse

def train(model, criterion, optimizer, batch_size, epoch, src_path, trg_path, sv_path):
    model.train()
    for e in range(epoch):
        print('Epoch {}'.format(e))
        for img_1, img_2, label in DataLoader(src_path, trg_path, batch_size, 'train'):
            optimizer.zero_grad()
            pred1, pred2 = model(img_1), model(img_2)
            loss = criterion(pred1, pred2, label)
            loss.backward()
            optimizer.step()
            print('Total loss for this batch: {}'.format(loss.item()))
        save_weights(model, e)

def save_weights(model, id):
    torch.save(model.state_dict(), 'weights/weights_1-19_{}.pt'.format(id+20))

def custom_loss_func(output, base_lf, target, temp=0.05):
    norm_out = F.normalize(output, dim=1)
    similarity = torch.matmul(norm_out, norm_out.T)
    return base_lf(similarity/temp, target)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/weights_1-19_130.pt', help='initial weights path')
    parser.add_argument('--data-source', type=str, default='./source', help='source image folder path')
    parser.add_argument('--data-target', type=str, default='./target', help='target image folder path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--learning-rate', type=int, default=1e-5, help='learning rate')
    #parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--save-dir', type=int, default='./weights', help='path to save weights')
    opt = parser.parse_args()
    
    model = models.maxvit_t().to('cuda') 
    model.load_state_dict(torch.load(opt.weights))
    optimize = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.7, weight_decay=1e-8)
    lf = nn.CrossEntropyLoss().to('cuda')
    loss_func = nn.CosineEmbeddingLoss(0.2)

    train(model, loss_func, optimize, opt.batch_size, opt.epochs, opt.data_source, opt.data_target, opt.save_dir)
    