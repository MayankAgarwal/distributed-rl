import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    
    def __init__(self, in_features, out_features, var0=0.4):
        
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.var0 = var0
        
        self.eps_w, self.eps_b = None, None
        
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.var_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_b = nn.Parameter(torch.Tensor(out_features))
        self.var_b = nn.Parameter(torch.Tensor(out_features))
        
        self.__init_eps()
        self.reset_parameters()
    
    def __init_eps(self):
        
        eps_i = torch.normal(
            means=torch.zeros(1, self.in_features), 
            std=torch.ones(1, self.in_features)
        )
        
        eps_j = torch.normal(
            means=torch.zeros(self.out_features, 1),
            std=torch.ones(self.out_features, 1)
        )
        
        f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
        f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))
        
        self.eps_w = Variable(torch.mm(f_eps_j, f_eps_i), requires_grad=False)
        self.eps_b = Variable(f_eps_j.squeeze(), requires_grad=False)
        
    def reset_parameters(self):
        
        std = 1.0/math.sqrt(self.in_features)
        
        self.mu_w.data.uniform_(-std, std)
        self.mu_b.data.uniform_(-std, std)
        
        self.var_w.data.fill_(self.var0*std)
        self.var_b.data.fill_(self.var0*std)
        
    def forward(self, input_):
        
        W = self.mu_w + self.var_w*self.eps_w
        b = self.mu_b + self.var_b*self.eps_b
        return F.linear(input_, W, b)
        