import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class LossNet(nn.Module):
    def __init__(self, kernel_size = 2, num_features=32, interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool1d(kernel_size)
        self.GAP2 = nn.AvgPool1d(kernel_size)
        self.GAP3 = nn.AvgPool1d(kernel_size)
        self.GAP4 = nn.AvgPool1d(kernel_size)


        self.FC1 = nn.Linear(num_features, interm_dim)
        self.FC2 = nn.Linear(num_features, interm_dim)
        self.FC3 = nn.Linear(num_features, interm_dim)
        self.FC4 = nn.Linear(num_features, interm_dim)
        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out