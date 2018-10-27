from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from bio_dataset import BioMixDataset
#from resnet import resnet20
from network import Regressor


parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default='./cleandata_1', help='path to dataset file')
parser.add_argument('--gen_file', default='./gan/fake_samples_epoch_009.npy', help='path to dataset file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0001')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--outf', default='./models', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop_ratio', type=float, default=0.2)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = BioMixDataset(opt.csv_file, opt.gen_file, split='train')
test_dataset = BioMixDataset(opt.csv_file, opt.gen_file, split='test')

assert train_dataset
assert test_dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset),
                                         shuffle=False, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)

net = Regressor(opt.drop_ratio).to(device)
        
if opt.net != '':
    net.load_state_dict(torch.load(opt.net))
print(net)

criterion = nn.MSELoss()

# setup optimizer
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)

#Train
best_epoch = -1
best_test_error = 1000000.
best_R_square = 0.

for epoch in range(opt.niter):
    net.train()
    for i, data in enumerate(train_dataloader, 0):
        net.zero_grad()
        x, y = data
        x = x.to(device)
        y = y.to(device)
        output = net(x)
        err = criterion(output, y)
        err.backward()
        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, opt.niter, i, len(train_dataloader),
                 err.item()))

    # do checkpointing
    torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))
#    if epoch >=100 and epoch < 150:
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = opt.lr * 0.1
#    elif epoch >= 150:
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = opt.lr * 0.01
            
    net.eval()
    err_sum = 0.
    R_square = 0.
    
    for i, data in enumerate(test_dataloader, 0):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)
        output = net(x)
        err = criterion(output, y)
        err_sum += float(err.item()) * batch_size
        
        y_hat = torch.sum(y)/ batch_size
        R_square = 1- float(err.item()) * batch_size/torch.sum((y - y_hat)*(y - y_hat)).item()
#        R_square = torch.sum((output - y_hat)*(output - y_hat))/torch.sum((y - y_hat)*(y - y_hat)).item()      
        
        
    test_err_ave = err_sum / len(test_dataset)
    print('R square: {}'.format(str(R_square)))
    print('test err: {}'.format(str(test_err_ave)))
    if abs(1-R_square) < abs(1-best_R_square):
        best_test_error = test_err_ave
        best_epoch = epoch
        best_R_square = R_square

print('best R square: {}'.format(str(best_R_square)))
print('best test err: %.4f' % (best_test_error))
print('best epoch: {}'.format(str(best_epoch)))