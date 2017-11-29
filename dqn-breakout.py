
# coding: utf-8

# In[1]:

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


# In[2]:

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import imageio
import gc
import os
# get_ipython().magic(u'matplotlib inline')


# In[3]:

from modules.dqn import DQN
from modules.env import Env
from modules.replay_memory import ReplayMemory

# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[4]:

print "PyTorch version: ", torch.__version__


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

GAME_ROM = 'roms/breakout.bin'

BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 100000  # Memory overflow with 10**6 sized replay memory
AGENT_HISTORY_LENGTH = 4
TARGET_NW_UPDATE_FREQ = 10**3  # 10**4
GAMMA = 0.99
ACTION_REPEAT = 4
IMG_RESCALE_SIZE = (84, 84)
PREFILL_REPLAY_MEM_STEPS = 50000
NOOP_RANGE = (0, 30)

EPS_START = 1
EPS_END = 0.1
FINAL_EPS_FRAME = 10**6

LR = 0.00005
REG = 0

TRAINING_STEPS = 500000
MODEL_SAVE_STEPS = 1000
MOVIE_SAVE_STEPS = 1000

RESULTS_FOLDER = 'results/dqn-breakout/'


# In[7]:

ENV = Env(
    os.path.abspath(GAME_ROM), IMG_RESCALE_SIZE, NOOP_RANGE, FloatTensor, AGENT_HISTORY_LENGTH, ACTION_REPEAT)

ACTIONS = ENV.action_set
ACTION_CNT = len(ACTIONS)

Transition = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state'))

dqn = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)
target_dqn = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)

if use_cuda:
    dqn.cuda()
    target_dqn.cuda()

optimizer = optim.RMSprop(dqn.parameters(), lr=LR, weight_decay=REG)
memory = ReplayMemory(REPLAY_MEMORY_SIZE, Transition)


# In[8]:

# Global variable definition

g_steps_done = 0
g_last_sync = 0
g_total_frames = 0


# In[9]:

def get_epsilon():
    global g_steps_done, g_total_frames
    
    if g_steps_done > FINAL_EPS_FRAME:
        return EPS_END

    eps = EPS_START + (EPS_END - EPS_START)*g_steps_done / FINAL_EPS_FRAME
    return eps

def select_action(state):
    
    global g_steps_done
    
    result = None
    rand = random.random()
    eps = get_epsilon()
    g_steps_done += 1
    
    if rand > eps:
        dqn.eval()  # Switch model to evaluation mode
        pred = dqn(Variable(state, volatile=True).type(FloatTensor)).data.max(1)
        dqn.train()  # Switch model back to train mode
        
        pred = pred[1].view(1, 1) # Single state action
        idx = int(pred[0].cpu().numpy())
        result = idx
    else:
        result = random.randrange(0, ACTION_CNT)
        
    return LongTensor([[result]])


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
    
    expected_state_action_values = reward_batch.squeeze() + GAMMA * next_state_values
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss.backward()
    
    for p in dqn.parameters():
        p.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    g_last_sync += 1


# In[11]:

def prefill_replay_mem():
    
    global memory
    
    done = False
    state = ENV.get_state()

    for i in xrange(PREFILL_REPLAY_MEM_STEPS):
        
        if done:
            ENV.reset_game()
            state = ENV.get_state()
            done = False

        action_idx = random.randrange(0, ACTION_CNT)
        action_val = ACTIONS[action_idx]
        action = LongTensor([[action_idx]])

        next_state, reward, done = ENV.take_action(action_val)
        reward = Tensor([[reward]])
        memory.push(state, action, reward, next_state)
        state = next_state


# In[12]:

game_rewards = []
total_rewards = 0.0
done = False
movie_frames = []

last_model_save, last_movie_save = 0, 0
save_movie = False

# TODO : Pre-fill experience replay memory
print "Filling experience replay memory with random actions"
prefill_replay_mem()
print "Replay memory initialized. Length = %d " % len(memory)

ENV.reset_game()
state = ENV.get_state()

print "Training starts"

for step_i in xrange(TRAINING_STEPS):
    
    if done:
        done = ENV.reset_game()
        state = ENV.get_state()
        
        print "Life complete. Reward = %f" % total_rewards
        game_rewards.append(total_rewards)
        
        if save_movie:
            imageio.mimsave(RESULTS_FOLDER + 'train-step_%d__eps_%f.gif' % (step_i, get_epsilon()), movie_frames)
            save_movie = False
        
        total_rewards = 0.0
        movie_frames = []
        
        gc.collect()
    
    action = select_action(state)
    action_val = ACTIONS[int(action.cpu().numpy()[0, 0])]
    
    next_state, reward, done = ENV.take_action(action_val)
    
    total_rewards += reward
    movie_frames.append(np.copy(ENV.get_current_screen()))
    reward = Tensor([[reward]])
    
    memory.push(state, action, reward, next_state)
    
    state = next_state
    optimize_model()
    
    if (step_i - last_model_save) >= MODEL_SAVE_STEPS:
        torch.save(dqn.state_dict(), RESULTS_FOLDER + 'dqn-%d.pth' % step_i)
        torch.save(target_dqn.state_dict(), RESULTS_FOLDER + 'target-dqn-%d.pth' % step_i)
        np.save(RESULTS_FOLDER + 'game_rewards', game_rewards)
        last_model_save = step_i
    
    if (step_i - last_movie_save) >= MOVIE_SAVE_STEPS:
        save_movie = True
        last_movie_save = step_i
        
    if step_i % 1000 == 0:
        print "%d steps trained. Epsilon: %f" % (step_i, get_epsilon())


# In[ ]:




# In[ ]:

"""
episode_rewards = []
w, h = ale.getScreenDims()

for episode in xrange(TRAIN_EPISODES):
    total_reward = 0.0
    done = False
    action_count = -1
    movie_frames = []
    
    if ale.lives() <= 0:
        ale.reset_game()
    
    temp_lives_rem = ale.lives()
    
    curr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    ale.getScreenRGB(screen_data=curr_frame)
    
    # Stores X+1 frames because it needs 0th index to compute the pixel max between 0th and 1st image
    last_K_frames = [curr_frame] * (AGENT_HISTORY_LENGTH+1)
    state = get_state_tensor(preprocessor.process_images(last_K_frames))
    
    # Use lives remaining to define an episode
    while not done:
        action_count = (action_count + 1)%ACTION_REPEAT
        
        # Skip-framing
        if action_count == 0:
            action = select_action(state)
            action_val = ACTIONS[int(action.cpu().numpy()[0, 0])]
        
        reward = ale.act(action_val)
        reward = max(min(1, reward), -1)  # Clamp rewards in range [-1, 1]
        done = (ale.lives() != temp_lives_rem)  # Consider life lost as a terminal state
        
        if not done:
            ale.getScreenRGB(screen_data=curr_frame)
            last_K_frames.append(curr_frame)
            last_K_frames = last_K_frames[:(AGENT_HISTORY_LENGTH+1)]
            next_state = get_state_tensor(preprocessor.process_images(last_K_frames))
        else:
            next_state = None
            
        total_reward += reward
        reward = Tensor([reward])
        g_total_frames += 1
            
        memory.push(state, action, reward, next_state)
        state = next_state
        optimize_model()
        
        if episode % SAVE_MOVIE_EVERY == 0: movie_frames.append(np.copy(curr_frame))
        
    if episode % SAVE_EVERY == 0:
        torch.save(dqn.state_dict(), RESULTS_FOLDER + 'dqn-%d.pth' % episode)
        torch.save(target_dqn.state_dict(), RESULTS_FOLDER + 'target-dqn-%d.pth' % episode)
        np.save(RESULTS_FOLDER + 'episode_rewards', episode_rewards)
    
    if episode % SAVE_MOVIE_EVERY == 0:
        imageio.mimsave(RESULTS_FOLDER + 'train-episode_%d__eps_%f.gif' % (episode, get_epsilon()), movie_frames)
    
    print "Episode %d, Total Reward: %f, Total frames: %f, eps: %f" % (episode, total_reward, g_total_frames, get_epsilon())
    episode_rewards.append(total_reward)
    
    gc.collect()
"""

