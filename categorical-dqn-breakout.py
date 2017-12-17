
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

from modules.categoricaldqn import CategoricalDQN
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
TARGET_NW_UPDATE_FREQ = 10**4
GAMMA = 0.99
ACTION_REPEAT = 4
IMG_RESCALE_SIZE = (84, 84)
PREFILL_REPLAY_MEM_STEPS = 1000 #50000
NOOP_RANGE = (0, 30)

EPS_START = 0.5
EPS_END = 0.1
FINAL_EPS_FRAME = 1e6

LR = 0.00025
REG = 0

TRAINING_STEPS = 5000000
MODEL_SAVE_STEPS = 25000
MOVIE_SAVE_STEPS = 25000

RESULTS_FOLDER = 'results/categorical_dqn-breakout/'


# In[7]:

V_min, V_max = -10., 10.
N_atoms = 51

delta_z = (V_max - V_min)/(N_atoms - 1)
support = torch.linspace(V_min, V_max, N_atoms).type(FloatTensor)


# In[8]:

ENV = Env(
    os.path.abspath(GAME_ROM), IMG_RESCALE_SIZE, NOOP_RANGE, FloatTensor, AGENT_HISTORY_LENGTH, ACTION_REPEAT)

ACTIONS = ENV.action_set
ACTION_CNT = len(ACTIONS)

Transition = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state'))

dqn = CategoricalDQN(AGENT_HISTORY_LENGTH, N_atoms, ACTION_CNT, is_noisy=False)
target_dqn = CategoricalDQN(AGENT_HISTORY_LENGTH, N_atoms, ACTION_CNT, is_noisy=False)

if use_cuda:
    dqn.cuda()
    target_dqn.cuda()

optimizer = optim.Adam(dqn.parameters(), lr=LR, weight_decay=REG)
memory = ReplayMemory(REPLAY_MEMORY_SIZE, Transition)


# In[9]:

# Global variable definition

g_steps_done = 0
g_last_sync = 0
g_total_frames = 0


# In[10]:

def get_Q_values(out_probs):
    global support
    # out_probs - (N, A, Z)
    support_cp = support.unsqueeze(1)  # Make support (Z, 1)
    q_values = torch.bmm(out_probs, support_cp.unsqueeze(0).expand(out_probs.size(0), *support_cp.size()).type(FloatTensor))
    q_values = q_values.squeeze()
    return q_values

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
    
    if rand >= eps:
        dqn.eval()  # Switch model to evaluation mode
        probs = dqn(Variable(state, volatile=True)).data  # (Actions x N_atoms)
        q_vals = get_Q_values(probs)
        pred = q_vals.max(0)   # Single state action selection
        dqn.train()  # Switch model back to train mode
        
        pred = pred[1].view(1, 1) # Single state action
        idx = int(pred[0].cpu().numpy())
        result = idx
    else:
        result = random.randrange(0, ACTION_CNT)
        
    return LongTensor([[result]])


# In[11]:

def optimize_model():
    global g_last_sync, support, V_min, V_max, delta_z, N_atoms
    
    dqn.zero_grad()
    
    if len(memory) < BATCH_SIZE:
        return
    
    # Sync target network with the prediction network
    if g_last_sync % TARGET_NW_UPDATE_FREQ == 0:
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/3
        target_dqn.load_state_dict(dqn.state_dict())
        target_dqn.zero_grad()
        
        for p in target_dqn.parameters():
            p.require_grad = False
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))  # 32 sized tensor (0-1 tensor)
    next_state_batch = Variable(torch.cat([s for s in batch.next_state
                                               if s is not None]), volatile=True)
    
    state_batch = Variable(torch.cat(batch.state))  # 32 x 4 x 84 x 84
    action_batch = Variable(torch.cat(batch.action)) # 32 x 1
    reward_batch = torch.cat(batch.reward) # 32 x 1
    
    p_s = dqn(state_batch)    # 32 x 4 x 51
    p_sa = p_s.gather(1, action_batch.unsqueeze(2).expand(BATCH_SIZE, 1, N_atoms))   # 32 x 1 x 51
    p_sa = p_sa.squeeze()   # 32 x 51
    
    # Double Q-learning
    s_t1_cnt = next_state_batch.size(0)
    p_s1 = dqn(next_state_batch).data  # 32 x 4 x 51
    q_s1 = get_Q_values(p_s1)   # 32 x 4
    a_opt_s1 = q_s1.max(1)[1].unsqueeze(1)   # 32 x 1 Long Tensor
    
    next_state_probs = target_dqn(next_state_batch).data   # X x 4 x 51
    next_state_action_probs = next_state_probs.gather(1, a_opt_s1.unsqueeze(2).expand(a_opt_s1.size(0), 1, N_atoms))  # X x 1 x 51
    next_state_action_probs = next_state_action_probs.squeeze()   # X x 51
    
    p_s1_a = torch.Tensor(BATCH_SIZE, N_atoms).zero_().type(FloatTensor)
    p_s1_a[non_final_mask.unsqueeze(1).expand(BATCH_SIZE, N_atoms)] = next_state_action_probs.type(FloatTensor)
    
    Tz = reward_batch + GAMMA * non_final_mask.unsqueeze(1).type(FloatTensor) * support.unsqueeze(0)  # (32, 51)
    
    Tz.clamp_(min=V_min, max=V_max)  # (32, 51)
    
    b = (Tz - V_min)/delta_z  # (32, 51)
    l, u = b.floor().long(), b.ceil().long()  # (32, 51)
    
    m = torch.Tensor(BATCH_SIZE, N_atoms).type(FloatTensor).zero_()
    offset = torch.linspace(0, ((BATCH_SIZE - 1) * N_atoms), BATCH_SIZE).long().unsqueeze(1).expand(BATCH_SIZE, N_atoms).type(LongTensor)
    m.view(-1).index_add_(0, (l + offset).view(-1), (p_s1_a * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (p_s1_a * (b - l.float())).view(-1))   # 32 x 51
    
    loss = -torch.sum(Variable(m, requires_grad=False) * p_sa.log())
    
    dqn.zero_grad()
    loss.backward()
    
    # for p in dqn.parameters():
    #     p.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    g_last_sync += 1


# In[ ]:

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


# In[ ]:

game_rewards = []
total_rewards = 0.0
done = False
movie_frames = []

last_model_save, last_movie_save = 0, 0

save_movie = False
save_frame = False

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
        
        if save_movie and save_frame:
            imageio.mimsave(RESULTS_FOLDER + 'train-step_%d__eps_%f.gif' % (step_i, get_epsilon()), movie_frames)
            save_movie = False
            save_frame = False
        elif save_movie and not save_frame:
            print "Saving movie"
            save_frame = True
        
        total_rewards = 0.0
        movie_frames = []
        
        gc.collect()
    
    action = select_action(state)
    action_val = ACTIONS[int(action.cpu().numpy()[0, 0])]
    
    next_state, reward, done = ENV.take_action(action_val)
    
    total_rewards += reward
    reward = Tensor([[reward]])
    
    if save_frame:
        movie_frames.append(np.copy(ENV.get_current_screen()))
    
    memory.push(state, action, reward, next_state)
    
    state = next_state
    optimize_model()
    
    if (step_i - last_model_save) >= MODEL_SAVE_STEPS:
        torch.save(dqn.state_dict(), RESULTS_FOLDER + 'categorical_dqn-%d.pth' % step_i)
        np.save(RESULTS_FOLDER + 'game_rewards', game_rewards)
        last_model_save = step_i
    
    if (step_i - last_movie_save) >= MOVIE_SAVE_STEPS:
        save_movie = True
        last_movie_save = step_i
        
    if step_i % 1000 == 0:
        print "%d steps trained. Epsilon: %f" % (step_i, get_epsilon())


# In[ ]:




# In[ ]:



