import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import NoisyLinear, NoisyFactorizedLinear

class DQN(nn.Module):
    
    def __init__(self, input_channels, action_cnt, is_noisy=False):
        
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        if is_noisy:
            self.fc4 = NoisyLinear(3136, 512)
            self.bn4 = nn.BatchNorm1d(512)
            self.out = NoisyLinear(512, action_cnt)
        else:
            self.fc4 = nn.Linear(3136, 512, bias=True)
            self.bn4 = nn.BatchNorm1d(512)
            self.out = nn.Linear(512, action_cnt, bias=True)
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc4(x)))
        
        return self.out(x)