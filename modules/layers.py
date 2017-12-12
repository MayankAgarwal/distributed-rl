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

        self.is_cuda = torch.cuda.is_available()
        self.floatTensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor
        
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.var_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_b = nn.Parameter(torch.Tensor(out_features))
        self.var_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def __init_eps(self):
        
        eps_i = torch.Tensor(self.in_features).normal_()
        eps_j = torch.Tensor(self.out_features).normal_()
        
        f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
        f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))
        
        eps_w = torch.ger(f_eps_j, f_eps_i)
        eps_b = f_eps_j

        return eps_w, eps_b
        
    def reset_parameters(self):
        
        std = 1.0/math.sqrt(self.in_features)
        
        self.mu_w.data.uniform_(-std, std)
        self.mu_b.data.uniform_(-std, std)
        
        self.var_w.data.fill_(self.var0*std)
        self.var_b.data.fill_(self.var0*std)
        
    def forward(self, input_):
        
        eps_w, eps_b = self.__init_eps()
        W = self.mu_w + self.var_w.mul(Variable(eps_w).type(self.floatTensor))
        b = self.mu_b + self.var_b.mul(Variable(eps_b).type(self.floatTensor))
        return F.linear(input_, W, b)
        