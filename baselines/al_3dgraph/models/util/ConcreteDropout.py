import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConcreteDropout(nn.Module):
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min_tensor = torch.tensor(init_min).to(device)
        init_max_tensor = torch.tensor(init_max).to(device)

        self.init_min = torch.log(init_min_tensor) - torch.log(1. - init_min_tensor)
        self.init_max = torch.log(init_max_tensor) - torch.log(1. - init_max_tensor)
        self.p_logit = torch.nn.Parameter(torch.empty(1).uniform_(self.init_min, self.init_max))
        self.p = torch.sigmoid(self.p_logit).to(device)

    def forward(self, x):
        output = self.concrete_dropout(x)
        return self.layer(output)

    def concrete_dropout(self, x):
        eps = 1e-07
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(torch.sigmoid(self.p_logit) + eps)
                     - torch.log(1. - torch.sigmoid(self.p_logit) + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1. - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - torch.sigmoid(self.p_logit)
        x = x * random_tensor / retain_prob
        return x

    def regularize(self):
        weight = torch.flatten(self.layer.weight)
        kr = self.weight_regularizer * torch.sum(weight**2) * (1. - torch.sigmoid(self.p_logit))
        dr = torch.sigmoid(self.p_logit) * torch.log(torch.sigmoid(self.p_logit)) + (1. - torch.sigmoid(self.p_logit)) * torch.log(1. - torch.sigmoid(self.p_logit))
        dr *= self.dropout_regularizer * weight.numel()
        return torch.sum(kr + dr)