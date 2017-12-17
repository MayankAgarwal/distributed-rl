import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

# class NoisyLinear(nn.Module):
    
#     def __init__(self, in_features, out_features, var0=0.4):
        
#         super(NoisyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.var0 = var0
#         self.eps_w, self.eps_b = None, None

#         self.is_cuda = torch.cuda.is_available()
#         self.floatTensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor
        
#         self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.var_w = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.mu_b = nn.Parameter(torch.Tensor(out_features))
#         self.var_b = nn.Parameter(torch.Tensor(out_features))
#         self.reset_parameters()
    
#     def __init_eps(self):
        
#         eps_i = torch.Tensor(self.in_features).normal_()
#         eps_j = torch.Tensor(self.out_features).normal_()
        
#         f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
#         f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))
        
#         eps_w = torch.ger(f_eps_j, f_eps_i)
#         eps_b = f_eps_j

#         return eps_w, eps_b
        
#     def reset_parameters(self):
        
#         std = 1.0/math.sqrt(self.in_features)
        
#         self.mu_w.data.uniform_(-std, std)
#         self.mu_b.data.uniform_(-std, std)
        
#         self.var_w.data.fill_(self.var0*std)
#         self.var_b.data.fill_(self.var0*std)
        
#     def forward(self, input_):
        
#         eps_w, eps_b = self.__init_eps()
#         W = self.mu_w + self.var_w.mul(Variable(eps_w).type(self.floatTensor))
#         b = self.mu_b + self.var_b.mul(Variable(eps_b).type(self.floatTensor))
#         return F.linear(input_, W, b)


# # https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)

        self.is_cuda = torch.cuda.is_available()
        self.floatTensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor

        sigma_init = sigma_zero / math.sqrt(in_features)

        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t()).type(self.floatTensor)
        
        noise_v = Variable(torch.mul(eps_in, eps_out)).type(self.floatTensor)
        
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

        
        
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.Tensor(out_features))
    self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
    self.register_buffer('bias_epsilon', torch.Tensor(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.weight_mu.size(1))
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

  def _scale_noise(self, size):
    x = torch.randn(size)
    x = x.sign().mul(x.abs().sqrt())
    return x

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(self._scale_noise(self.out_features))

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon)), self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon)))
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)