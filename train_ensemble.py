import torch
import torch.optim as optim
import torch.nn.functional as F

from DataLoaderCombo import DataLoaderCombo
from ModelCombo import ModelCombo

from match import *
from tqdm.notebook import trange

import argparse

def train(model, optimizer, batch_size, epoch, src_path, trg_path, sv_path):
  model.train()
  with open('loss_epoch.txt', 'w') as f:
    with open('eval_score.txt', 'w') as es:
      for e in trange(epoch):
        losses = []
        print('Epoch {}'.format(e))
        for img, label in DataLoaderCombo(src_path, trg_path, batch_size, augments=True):
            
            if img.shape[0] != batch_size*2:
              continue
            
            optimizer.zero_grad()
            pred =  model(img)
            loss = contrast_loss_func(pred, label, 0.05)
            loss.backward()
            optimizer.step()
            print('Total loss for this batch: {}'.format(loss.item()))
            losses.append(loss.item())
        save_weights(model, e, sv_path)
        avg_loss = sum(losses)/len(losses)
        print('Average Loss for Epoch: {}'.format(avg_loss))
        f.write('{}\n'.format(avg_loss))

        #eval
        if e > -1:
          acc = match(model, src_path, trg_path, 15, '')
          print('Accuracy for this epoch: {}'.format(acc))
          es.write('{}\n'.format(acc))
        
        

def contrast_loss_func(output, target, temp=0.05):
    norm_out = F.normalize(output, dim=1)
    similarity = torch.matmul(norm_out, norm_out.T)/temp
    
    similarity = similarity.fill_diagonal_(-float('inf'))
    neg = torch.sum(torch.exp(similarity), dim=-1)

    N = similarity.shape[0]
    pos = torch.exp(similarity[torch.arange(N), target])
    loss = -torch.log(pos/neg).mean()

    return loss

def save_weights(model, id, save_path):
    state_dict =  model.state_dict()
    keys = state_dict.copy().keys()
    for key in keys:
        if 'model' in key:
            del state_dict[key]
    torch.save(state_dict, '{}/weights_2-27_{}.pt'.format(save_path, id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/weights_1-19_130.pt', help='initial weights path')
    parser.add_argument('--source', type=str, default='./source', help='source image folder path')
    parser.add_argument('--target', type=str, default='./target', help='target image folder path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (L2 penalty)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--save-dir', type=str, default='./weights', help='path to save weights')
    opt = parser.parse_args()
    
    model = ModelCombo().to('cuda')
    model.load_state_dict(torch.load(opt.weights), strict=False)

    for name, param in model.named_parameters():
        if param.requires_grad and 'model' in name:
            param.requires_grad = False
    
    nonfrozen_params = [param for param in model.parameters() if param.requires_grad]

    #learn_rate = opt.batch_size/256*0.3

    optimize = optim.SGD(nonfrozen_params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    #optimizer = torch.optim.AdamW()
    train(model, optimize, opt.batch_size, opt.epochs, opt.source, opt.target, opt.save_dir)