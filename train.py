import torch
import torch.optim as optim
import torch.nn.functional as F

from DataLoader import DataLoader
from ModelCombo import ModelCombo

from match import *
from tqdm import tqdm

import argparse


def train(model, optimizer, batch_size, epoch, src_path, trg_path, sv_path, eval, metrics):
    model.train()

    log_avg_loss = []
    log_eval_acc = []

    for e in tqdm(range(epoch)):
        losses = []
        print('\nEpoch {}'.format(e+1))
        for img, label in DataLoader(src_path, trg_path, batch_size, augments=True):

            if img.shape[0] != batch_size*2:
                continue

            optimizer.zero_grad()
            pred = model(img)
            loss = contrast_loss_func(pred, label, 0.05)
            loss.backward()
            optimizer.step()
            print('Total loss for this batch: {}'.format(loss.item()))
            losses.append(loss.item())

        save_weights(model, e+1, sv_path)
        avg_loss = sum(losses)/len(losses)
        print('Average Loss for Epoch: {}'.format(avg_loss))
        log_avg_loss.append(avg_loss)

        # eval
        if e > -1 and eval:
            eval_acc = match(model, src_path, trg_path, 15, '')
            print('Accuracy for this epoch: {}'.format(eval_acc))
            log_eval_acc.append(eval_acc)

    if metrics:
        with open(metrics + 'avgLoss.txt', 'w') as lossfile, open(metrics + 'evalAcc.txt', 'w') as evalfile:
            lossfile.write('{}\n'.format())
            evalfile.write('{}\n'.format())


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
    state_dict = model.state_dict()
    keys = state_dict.copy().keys()

    for key in keys:
        if 'model' in key:
            del state_dict[key]

    torch.save(state_dict, '{}/{}.pt'.format(save_path, id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='', help='initial weights path')
    parser.add_argument('--source', type=str,
                        default='./source', help='source image folder path')
    parser.add_argument('--target', type=str,
                        default='./target', help='target image folder path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-3, help='weight decay (L2 penalty)')
    parser.add_argument('--dropout', type=float,
                        default=0.3, help='dropout probability while training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str,
                        default='./weights', help='path to save weights')
    parser.add_argument('--metrics', type=str,
                        default='', help='Save avg loss and eval accuracy to separate files within given folder')
    parser.add_argument('--eval', type=bool,
                        default=False, help='Run matching after every epoch')
    # parser.add_argument('--freeze', type=bool,
    #                    default=False, help='Freeze backbone weights')
    opt = parser.parse_args()

    model = ModelCombo().to(opt.device)

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights), strict=False)

    # if opt.freeze:
    for name, param in model.named_parameters():
        if param.requires_grad and 'model' in name:
            param.requires_grad = False

    nonfrozen_params = [
        param for param in model.parameters() if param.requires_grad]

    optimize = optim.SGD(nonfrozen_params, lr=opt.learning_rate,
                         momentum=0.9, weight_decay=opt.weight_decay)

    # optimizer = torch.optim.Adam(nonfrozen_params, lr=opt.learning_rate)

    train(model, optimize, opt.batch_size, opt.epochs,
          opt.source, opt.target, opt.save_dir, eval=opt.eval, metrics=opt.metrics)
