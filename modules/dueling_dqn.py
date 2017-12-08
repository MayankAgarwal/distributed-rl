import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    
    def __init__(self, input_channels, action_cnt):
        
        super(DuelingDQN, self).__init__()
        
        self.action_cnt = action_cnt
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Value function FC stream
        self.val_fc1 = nn.Linear(3136, 512, bias=True)
        self.val_bn1 = nn.BatchNorm1d(512)
        
        self.val_out = nn.Linear(512, 1, bias=True)
        
        # Advantage function FC stream
        self.advantage_fc1 = nn.Linear(3136, 512, bias=True)
        self.advantage_bn1 = nn.BatchNorm1d(512)
        
        self.advantage_out = nn.Linear(512, action_cnt, bias=True)
    
        
    def forward(self, x):
        
        # Compute image features using common CONV layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        # Compute state-value function
        val_x = F.relu(self.val_bn1(self.val_fc1(x)))
        state_value = self.val_out(val_x)
        
        # Compute advantage function
        adv_x = F.relu(self.advantage_bn1(self.advantage_fc1(x)))
        advantage = self.advantage_out(adv_x)
        
        # Compute Q-value
        q_val = state_value + (advantage - advantage.sum()/self.action_cnt)
        
        return q_val