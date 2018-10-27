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

from bio_dataset import BioDataset


parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default='./cleandata_1', help='path to dataset file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=20, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./gan', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--alpha', type=float, default=1.)

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

dataset = BioDataset(opt.csv_file,split='train')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 
            nn.Linear(nz+1, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # 
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # 
            nn.Linear(512, 20),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(20, 512),
#            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 512),
#            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 512),
#            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
#            nn.Linear(512, 1),
#            nn.Sigmoid()
        )
        self.L1 = nn.Linear(512,1)
        self.L2 = nn.Linear(512,1)
        self.critic = nn.Sequential(self.main, self.L1)
        self.regressor = nn.Sequential(self.main, self.L2, nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.critic, input, range(self.ngpu))
            reg_output = nn.parallel.data_parallel(self.regressor, input, range(self.ngpu))
        else:
            output = self.critic(input)
            reg_output = self.regressor(input)

        return output.view(-1, 1).squeeze(1), reg_output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.MSELoss()

fixed_noise = torch.rand(len(dataset), nz+1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        real_label = real_cpu[:,-1]
        real_cpu = real_cpu[:,:-1]
        output_real, reg_output_real = netD(real_cpu)
        d_real_loss = -output_real.mean()
        d_real_reg_loss = criterion(reg_output_real, real_label)

        # train with fake
        noise = torch.rand(batch_size, nz+1, device=device)
        fake = netG(noise)
        
        output_fake, reg_output_fake = netD(fake.detach())
        d_fake_loss = output_fake.mean()

        #gradient penalty
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        x_hat = (alpha * real_cpu.data + (1 - alpha) * fake.data).requires_grad_(True)
        out_hat, reg_out_hat = netD(x_hat)
        weight = torch.ones(out_hat.size()).to(device)
        dydx = torch.autograd.grad(outputs=out_hat,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))        
        d_loss_gp = torch.mean((dydx_l2norm-1)**2) 
        
        d_loss = d_real_loss + d_fake_loss + opt.alpha*d_real_reg_loss + opt.lamda * d_loss_gp
        d_loss.backward()
        optimizerD.step()
        print('[%d/%d][%d/%d] loss_d: %.4f, d_real_cost: %.4f, d_fake_cost: %.4f, d_real_reg_loss: %.4f, d_gp: %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     d_loss.item(), -d_real_loss.item(), d_fake_loss.item(), d_real_reg_loss.item(),d_loss_gp.item()))

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % opt.n_critic == 0:
            netG.zero_grad()
            noise = torch.rand(batch_size, nz+1, device=device)
            fake = netG(noise)
            fake_label = noise[:,-1]
            
            output_fake, reg_output_fake = netD(fake)
            g_fake_loss = -output_fake.mean()
            g_reg_loss = criterion(reg_output_fake, fake_label)
            g_loss = g_fake_loss + opt.alpha*g_reg_loss
            g_loss.backward()
            optimizerG.step()
    
            print('[%d/%d][%d/%d] loss_g: %.4f, g_fake_cost: %.4f, g_reg_loss: %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     g_loss.item(), -g_fake_loss.item(), g_reg_loss.item()))
        if i % 37 == 0:
            netG.eval()
            fake = netG(fixed_noise).detach().cpu().numpy()
            fake_label = fixed_noise[:,-1:].cpu().numpy()
            fake = np.concatenate((fake,fake_label), axis=1)
            np.save('%s/fake_samples_epoch_%03d' % (opt.outf, epoch), fake)
#            vutils.save_image(fake.detach(),
#                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
#                    normalize=True)
            netG.train()

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))