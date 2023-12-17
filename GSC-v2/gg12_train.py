import os
import time
import json
import argparse
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from functools import partial
import pandas as pd

import data
from load_gg12 import GCommandLoader

from spiking_neuron.neuron import LIFNode
from spiking_neuron.tclif import TCLIFNode
from models.fc import ffGSC, fbGSC
from utils import *


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()
    for i, (text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        text = text.cuda(non_blocking=True)  # text:[bs, 1, 80, T]
        target = target.cuda(non_blocking=True)

        text_in = text.squeeze(1).permute(2, 0, 1)  # text:[T, bs, 80]
        reset_states(model=model)
        output = model(text_in)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), text.size(0))
        top1.update(acc1[0], text.size(0))
        top5.update(acc5[0], text.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                top1.avg))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (text, target) in enumerate(val_loader):
            text = text.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            text_in = text.squeeze(1).permute(2, 0, 1)  # text:[T, bs, 80]

            reset_states(model=model)
            output = model(text_in)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), text.size(0))
            top1.update(acc1[0], text.size(0))
            top5.update(acc5[0], text.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


parser = argparse.ArgumentParser(description='GSC 12 Keyword Spotting')
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')  # sgd
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--schedule', default=[30, 60, 90, 120], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')

# options for SNNs
parser.add_argument('--time-window', default=101, type=int, help='')
parser.add_argument('--threshold', default=1.0, type=float, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--hard-reset', action='store_true', default=False, help='')
parser.add_argument('--decay-factor', default=1.0, type=float, help='')
parser.add_argument('--beta_init1', default=0., type=float, help='')
parser.add_argument('--beta_init2', default=0., type=float, help='')
parser.add_argument('--gamma', default=0.7, type=float, help='dendritic reset scaling hyper-parameter')
parser.add_argument('--sg', default='triangle', type=str, help='surrogate gradient: triangle and exp ')
parser.add_argument('--neuron', default='tclif', type=str, help='tclif, lif')
parser.add_argument('--network', default='fb', type=str, help='fb, ff')
parser.add_argument('--drop', default=0.3, type=float, help='')
parser.add_argument('--version', default='v2', type=str)

args = parser.parse_args()

data.google12_v2(version=args.version)

if args.results_dir == '':
    args.results_dir = './cs-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

Path(args.results_dir).mkdir(parents=True, exist_ok=True)
logger = setup_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

gpu = torch.device('cuda')
seed_everything(seed=args.seed, is_cuda=True)

torch.backends.cudnn.benchmark = True

# build dataloaders
cwd = os.getcwd()
if args.version == 'v1':
    train_dataset = GCommandLoader(cwd+'/../data/google_speech_command_1/processed/train',
                                   window_size=.02, max_len=args.time_window)
    test_dataset = GCommandLoader(cwd+'/../data/google_speech_command_1/processed/test',
                                  window_size=.02, max_len=args.time_window)
elif args.version == 'v2':
    train_dataset = GCommandLoader(cwd+'/../data/google_speech_command_2/processed/train',
                                   window_size=.02, max_len=args.time_window)
    test_dataset = GCommandLoader(cwd+'/../data/google_speech_command_2/processed/test',
                                  window_size=.02, max_len=args.time_window)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory='cpu', sampler=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=None, num_workers=8, pin_memory='cpu', sampler=None)

if args.sg == 'exp':
    from surrogate import SingleExponential as SG
elif args.sg == 'triangle':
    from surrogate import Triangle as SG
elif args.sg == 'rectangle':
    from surrogate import Rectangle as SG
else:
    raise NotImplementedError

node = None
if args.neuron == 'lif':
    node = LIFNode
elif args.neuron == 'tclif':
    node = TCLIFNode

# initialize the learnable betas
beta = torch.full([1, 2], 0, dtype=torch.float)
beta[0][0] = args.beta_init1
beta[0][1] = args.beta_init2
init1 = torch.sigmoid(beta[0][0]).cpu().item()
init2 = torch.sigmoid(beta[0][1]).cpu().item()
print("beta init from {:.2f} and {:.2f}".format(-init1, init2))

spk_params = {"time_window": args.time_window,
              'v_threshold': args.threshold,
              'surrogate_function': SG.apply,
              'hard_reset': args.hard_reset,
              'detach_reset': False,
              'decay_factor': beta,
              'gamma': args.gamma}

spiking_neuron = partial(node,
                         v_threshold=spk_params['v_threshold'],
                         surrogate_function=spk_params['surrogate_function'],
                         hard_reset=spk_params['hard_reset'],
                         detach_reset=spk_params['detach_reset'],
                         decay_factor=spk_params['decay_factor'],
                         gamma=spk_params['gamma'])

in_dim = 40
if args.network == 'ff':
    model = ffGSC(in_dim=in_dim, spiking_neuron=spiking_neuron, drop=args.drop).to(gpu)
elif args.network == 'fb':
    model = fbGSC(in_dim=in_dim, spiking_neuron=spiking_neuron, drop=args.drop).to(gpu)

logging.info(str(model))
para_num = count_parameters(model)

criterion = nn.CrossEntropyLoss().cuda(gpu)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise NotImplementedError

# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
logging.info(str(args))
start_epoch = 0

if args.print_freq > len(train_loader):
    args.print_freq = math.ceil(len(train_loader) // 2)

best_acc = argparse.Namespace(top1=0, top5=0)
# For storing results
train_res = pd.DataFrame()
test_res = pd.DataFrame()
best = 0
for epoch in range(start_epoch, args.epochs):
    flag = False
    adjust_learning_rate(optimizer, epoch, args)

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

    acc1, acc5 = validate(test_loader, model, criterion, args)
    best_acc.top1 = max(best_acc.top1, acc1)
    best_acc.top5 = max(best_acc.top5, acc5)
    train_res[str(epoch)] = [train_acc.cpu().item(), train_loss]
    test_res[str(epoch)] = [acc1.cpu().item()]

    if acc1.cpu().item() >= best:
        flag = True
        best = acc1.cpu().item()
    print('Test Epoch: [{}/{}], lr: {:.6f}, acc: {:.4f}, best: {:.4f}'.format(epoch, args.epochs,
                                                                       optimizer.param_groups[0]['lr'],
                                                                       acc1, best))

    train_res.to_csv(os.path.join(args.results_dir, 'train_res.csv'), index=False)
    test_res.to_csv(os.path.join(args.results_dir, 'test_res.csv'), index=False)
    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=flag, dirname=args.results_dir, filename='checkpoint.pth.tar')
