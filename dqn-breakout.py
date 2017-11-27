
# coding: utf-8

# In[1]:

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from ale_python_interface import ALEInterface


# In[2]:

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import imageio
# get_ipython().magic(u'matplotlib inline')


# In[3]:

from modules.dqn import DQN
from modules.preprocess import Preprocess
from modules.replay_memory import ReplayMemory

# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[4]:

GAME_ROM = 'roms/breakout.bin'

ale = ALEInterface()
ale.setBool('display_screen', False)
ale.loadROM(GAME_ROM)


# In[5]:

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Refer https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
# Since the input is fixed size, this flag could be set to True on GPU for faster performance.
torch.backends.cudnn.benchmark = use_cuda


# In[6]:

BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10**6
AGENT_HISTORY_LENGTH = 4
TARGET_NW_UPDATE_FREQ = 10**4
GAMMA = 0.99
ACTION_REPEAT = 4
IMG_RESCALE_SIZE = (84, 84)

EPS_START = 1
EPS_END = 0.1
FINAL_EPS_FRAME = 10**6

LR = 0.00025
GRADIENT_MOMENTUM = 0.95    # UNUSED
SQ_GRADIENT_MOMENTUM = 0.95 # UNUSED

ACTIONS = ale.getLegalActionSet()
ACTION_CNT = len(ACTIONS)

TRAIN_EPISODES = 100001
SAVE_EVERY = 10
SAVE_MOVIE_EVERY = 50


# In[7]:

preprocessor = Preprocess(IMG_RESCALE_SIZE)
Transition = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state'))

dqn = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)
target_dqn = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)

if use_cuda:
    dqn.cuda()
    target_dqn.cuda()

optimizer = optim.RMSprop(dqn.parameters(), lr=LR)
memory = ReplayMemory(REPLAY_MEMORY_SIZE, Transition)


# In[8]:

# Global variable definition

g_steps_done = 0
g_last_sync = 0


# In[9]:

def get_epsilon():
    global g_steps_done
    
    if g_steps_done > FINAL_EPS_FRAME:
        return EPS_END

    eps = EPS_START + (EPS_END - EPS_START)*g_steps_done / FINAL_EPS_FRAME
    return eps

def select_action(state):
    
    global g_steps_done
    
    rand = random.random()
    eps = get_epsilon()
    g_steps_done += 1
    
    if rand > eps:
        pred = dqn(Variable(state, volatile=True).type(FloatTensor)).data.max(1)
        pred = pred[1].view(1, 1) # Single state action
        
        idx = int(pred[0].cpu().numpy())
        act = LongTensor([[int(ACTIONS[idx])]])
        return act
    else:
        return LongTensor([[int(random.choice(ACTIONS))]])

def get_state_tensor(state):
    
    state_tensor = torch.from_numpy(state).type(FloatTensor)
    if len(state_tensor.size()) == 3:  # 3d tensor
        state_tensor = state_tensor.unsqueeze(0)
    
    return state_tensor


# In[10]:

def optimize_model():
    global g_last_sync
    
    dqn.zero_grad()
    
    if len(memory) < BATCH_SIZE:
        return
    
    if g_last_sync % TARGET_NW_UPDATE_FREQ == 0:
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/3
        target_dqn.load_state_dict(dqn.state_dict())
        target_dqn.zero_grad()
        
        for p in target_dqn.parameters():
            p.require_grad = False
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                               if s is not None]), volatile=True)
    
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    
    state_action_values = dqn(state_batch).gather(1, action_batch)
    
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0]
    
    next_state_values.volatile = False
    
    expected_state_action_values = reward_batch + GAMMA * next_state_values
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss.backward()
    optimizer.step()
    
    g_last_sync += 1


# In[11]:

episode_rewards = []
total_frames = 0

for episode in xrange(TRAIN_EPISODES):
    total_reward = 0.0
    done = False
    action_count = 0
    temp_lives_rem = -1
    movie_frames = []
    
    ale.reset_game()
    w, h = ale.getScreenDims()
    curr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    ale.getScreenRGB(screen_data=curr_frame)
    
    # Stores X+1 frames because it needs 0th index to compute the pixel max between 0th and 1st image
    last_K_frames = [curr_frame] * (AGENT_HISTORY_LENGTH+1)
    state = get_state_tensor(preprocessor.process_images(last_K_frames))
    
    # Use lives remaining to define an episode
    while not done:
        # Skip-framing
        if action_count == 0:
            action = select_action(state)
            action_int = int(action.cpu().numpy()[0, 0])
        
        action_count = (action_count + 1)%ACTION_REPEAT
        
        reward = ale.act(action_int)
        reward = max(min(1, reward), -1)  # Clamp rewards in range [-1, 1]
        done = (ale.lives()<=0)
        
        if episode % SAVE_MOVIE_EVERY == 0: movie_frames.append(np.copy(curr_frame))
        
        total_reward += reward
        reward = Tensor([reward])
        total_frames += 1e-6
        
        if ale.lives() != temp_lives_rem:
            print "\t Lives remaining: ", ale.lives()
            temp_lives_rem = ale.lives()
        
        if not done:
            ale.getScreenRGB(screen_data=curr_frame)
            last_K_frames.append(curr_frame)
            last_K_frames = last_K_frames[:(AGENT_HISTORY_LENGTH+1)]
            next_state = get_state_tensor(preprocessor.process_images(last_K_frames))
        else:
            next_state = None
            
        memory.push(state, action, reward, next_state)
        state = next_state
        optimize_model()
        
    if episode % SAVE_EVERY == 0:
        torch.save(dqn.state_dict(), 'results/dqn-%d.pth' % episode)
        torch.save(target_dqn.state_dict(), 'results/target-dqn-%d.pth' % episode)
        np.save('results/episode_rewards', episode_rewards)
    
    if episode % SAVE_MOVIE_EVERY == 0:
        imageio.mimsave('results/train-episode_%d__eps_%f.gif' % (episode, get_epsilon()), movie_frames)
    
    print "Episode %d, Total Reward: %f, Total frames: %f, eps: %f" % (episode, total_reward, total_frames, get_epsilon())
    episode_rewards.append(total_reward)


# In[ ]:




# In[ ]:



