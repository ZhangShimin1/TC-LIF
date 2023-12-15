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
from torch.optim.lr_scheduler import StepLR
from spiking_neuron.neuron import LIFNode
from spiking_neuron.PLIF import ParametricLIFNode
from spiking_neuron.TCLIF import TCLIFNode
from load_dataset import getData
from models.fc import ff_SHD, fb_SHD_v1
from utils import *


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)  # images:[bs, 1, 28, 28]
        target = target.cuda(non_blocking=True)

        reset_states(model=model)
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        #    progress.display(i)
    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                top1.avg))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            reset_states(model=model)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # torch.cuda.synchronize()

            # if i % 10 == 0:
            #     progress.display(i)

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

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


parser = argparse.ArgumentParser(description='Sequential SHD/SSC')
parser.add_argument('--task', default='SHD', type=str, help='SHD, SSC')
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--schedule', default=[40, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument("--workers", type=int, default=8)
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')

# options for SNNs
parser.add_argument('--time-window', default=250, type=int, help='')
parser.add_argument('--threshold', default=1.5, type=float, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--hard-reset', action='store_true', default=False, help='')
parser.add_argument('--decay-factor', default=1.0, type=float, help='')
parser.add_argument('--beta1', default=0., type=float, help='')
parser.add_argument('--beta2', default=0., type=float, help='')
parser.add_argument('--gamma', default=0.5, type=float, help='dendritic reset scaling hyper-parameter')
parser.add_argument('--sg', default='triangle', type=str, help='surrogate gradient: triangle and exp')
parser.add_argument('--neuron', default='tclif', type=str, help='plif, lif, tclif')
parser.add_argument('--network', default='ff', type=str, help='fb, ff')

args = parser.parse_args()
if args.results_dir == '':
    args.results_dir = './exp/'+args.task+args.network+'-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

Path(args.results_dir).mkdir(parents=True, exist_ok=True)
logger = setup_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')
gpu = torch.device('cuda')
seed_everything(seed=args.seed, is_cuda=True)

torch.backends.cudnn.benchmark = True

# build dataloaders
class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.int)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                device)
            y_batch = torch.tensor(self.labels_[batch_index], device=device)
            self.counter += 1
            return X_batch.to(device=device), y_batch.to(device=device)

        else:
            raise StopIteration

# get datasets and build dataloaders
if args.task == 'SHD' or args.task == 'SSC':
    T = 250
    max_time = 1.4
    in_dim = 700
    (x_train, y_train), (x_test, y_test) = getData(args.task)
    train_loader = SpikeIterator(x_train, y_train, args.batch_size, T, in_dim, max_time, shuffle=True)
    test_loader = SpikeIterator(x_test, y_test, args.batch_size, T, in_dim, max_time, shuffle=False)
else:
    raise NotImplementedError

if args.sg == 'exp':
    from surrogate import SingleExponential as SG
elif args.sg == 'triangle':
    from surrogate import Triangle as SG
elif args.sg == 'rectangle':
    from surrogate import Rectangle as SG
else:
    raise NotImplementedError

node = None
if args.neuron == 'plif':
    node = ParametricLIFNode
elif args.neuron == 'lif':
    node = LIFNode
elif args.neuron == 'tclif':
    node = TCLIFNode

# initialize the learnable betas
beta = torch.full([1, 2], 0., dtype=torch.float)
beta[0][0] = args.beta1
beta[0][1] = args.beta2
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

if args.task == 'SHD':
    if args.network == 'ff':
        model = ff_SHD(in_dim=in_dim, hidden=128, out_dim=20, spiking_neuron=spiking_neuron).to(gpu)
    elif args.network == 'fb':
        model = fb_SHD_v1(in_dim=in_dim, hidden=128, out_dim=20, spiking_neuron=spiking_neuron).to(gpu)
elif args.task == 'SSC':
    if args.network == 'ff':
        model = ff_SHD(in_dim=in_dim, hidden=400, out_dim=35, spiking_neuron=spiking_neuron).to(gpu)
    elif args.network == 'fb':
        model = fb_SHD_v1(in_dim=in_dim, hidden=400, out_dim=35, spiking_neuron=spiking_neuron).to(gpu)
else:
    raise NotImplementedError

logging.info(str(model))

criterion = nn.CrossEntropyLoss().cuda(gpu)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise NotImplementedError
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

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
for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args)

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
    train_loader.reset()
    acc1, acc5 = validate(test_loader, model, criterion, args)
    test_loader.reset()
    #scheduler.step()

    best_acc.top1 = max(best_acc.top1, acc1)
    best_acc.top5 = max(best_acc.top5, acc5)
    train_res[str(epoch)] = [train_acc.cpu().item(), train_loss]
    test_res[str(epoch)] = [acc1.cpu()]

    logging.info('Test Epoch: [{}/{}], lr: {:.6f}, acc: {:.4f}'.format(epoch, args.epochs,
                                                                       optimizer.param_groups[0]['lr'],
                                                                       acc1))
    logging.info('Best test acc: {:.4f}'.format(best_acc.top1))
    #print('current beta values:', model)

    train_res.to_csv(os.path.join(args.results_dir, 'train_res.csv'), index=False)
    test_res.to_csv(os.path.join(args.results_dir, 'test_res.csv'), index=False)
    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, dirname=args.results_dir, filename='checkpoint.pth.tar')
