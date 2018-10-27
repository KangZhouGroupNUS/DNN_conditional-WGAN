import torch
import torch.nn as nn
import math

class CNNRegressor(nn.Module):
    def __init__(self, nf):
        super(Regressor, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, nf, 5, 1, 0, bias=False),
            nn.BatchNorm1d(nf),
            nn.ReLU(inplace=True),
            #nf x 16
            nn.Conv1d(nf, 2*nf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(2*nf),
            nn.ReLU(inplace=True),
            #2nf x 8
            nn.Conv1d(2*nf, 4*nf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(4*nf),
            nn.ReLU(inplace=True),
            #4nf x 4
            nn.Conv1d(4*nf, 1, 4, 1, 0)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.main(x)
        return torch.squeeze(torch.squeeze(x, dim=1), dim=1)
        
class Regressor(nn.Module):
    def __init__(self, drop_ratio = 0.2):
        super(Regressor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(20, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_ratio),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_ratio),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return torch.squeeze(self.main(x), dim=1)