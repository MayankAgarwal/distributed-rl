
# coding: utf-8

# In[1]:

import torch
from torch.autograd import Variable


# In[2]:

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import imageio
# get_ipython().magic(u'matplotlib inline')


# In[3]:

from modules.dqn import DQN
from modules.dueling_dqn import DuelingDQN
from modules.categoricaldqn import CategoricalDQN
from modules.env import Env

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

MODEL_TYPE = "cdqn"
MODELS_PATH = sys.argv[1] #'results/dqn-breakout'
GAME_ROM = 'roms/breakout.bin'
PER_MODEL_PLAYS = 10

BATCH_SIZE = 32
AGENT_HISTORY_LENGTH = 4
ACTION_REPEAT = 4
IMG_RESCALE_SIZE = (84, 84)
EPS = 0.05
NOOP_RANGE = (0, 0)
N_atoms = 51

RESULTS_FOLDER = sys.argv[2] #'results/[T] dqn_play/'


# In[ ]:

V_min, V_max = -10., 10.
N_atoms = 51

delta_z = (V_max - V_min)/(N_atoms - 1)
support = torch.linspace(V_min, V_max, N_atoms)


# In[7]:

try:
    os.makedirs(RESULTS_FOLDER)
except Exception as _:
    pass


# In[8]:

ENV = Env(
    os.path.abspath(GAME_ROM), IMG_RESCALE_SIZE, NOOP_RANGE, FloatTensor, AGENT_HISTORY_LENGTH, ACTION_REPEAT)

ACTIONS = ENV.action_set
ACTION_CNT = len(ACTIONS)

if MODEL_TYPE == 'dqn':
    DQN = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)
elif MODEL_TYPE == 'ddqn':
    DQN = DQN(AGENT_HISTORY_LENGTH, ACTION_CNT)
elif MODEL_TYPE == 'dueldqn':
    DQN = DuelingDQN(AGENT_HISTORY_LENGTH, ACTION_CNT)
elif MODEL_TYPE == 'cdqn':
    DQN = CategoricalDQN(AGENT_HISTORY_LENGTH, N_atoms, ACTION_CNT)

if use_cuda:
    DQN.cuda()


# In[9]:

def get_model_list():
    global MODELS_PATH
    models = []
    model_files = filter(lambda f: f.endswith('.pth'), os.listdir(MODELS_PATH))
    
    for i, f in enumerate(model_files):
        iteration = int(f.split('-')[1].split('.')[0])
        if i==0 or iteration%25000 == 0:
            models.append((iteration, os.path.join(MODELS_PATH, f)))
    models.sort(key=lambda x: x[0])
    
    return models

def load_model(model_filepath):
    global DQN
    DQN.load_state_dict(torch.load(model_filepath, map_location=lambda storage, loc: storage))
    DQN.eval()


# In[10]:

def get_Q_values(out_probs):
    global support
    # out_probs - (N, A, Z)
    support_cp = support.unsqueeze(1)  # Make support (Z, 1)
    q_values = torch.bmm(out_probs, support_cp.unsqueeze(0).expand(out_probs.size(0), *support_cp.size()).type(FloatTensor))
    q_values = q_values.squeeze()
    return q_values

def select_action(state):
    
    global EPS, DQN
    
    result = None
    rand = random.random()
    
    if rand < EPS:
        result = random.randrange(0, ACTION_CNT)
    else:
        pred = DQN(Variable(state, volatile=True).type(FloatTensor)).data
        qvals = get_Q_values(pred)
        pred = qvals.max(0)
        pred = pred[1].view(1, 1)
        idx = int(pred[0].cpu().numpy())
        result = idx
        
    return result


# In[11]:

def play_game(save_movie=False, movie_name=None):
    total_reward = 0.0
    done = False
    movie_frames = []
    ENV.reset_game()
    
    while not done:
        state = ENV.get_state()
        action_idx = select_action(state)
        action = ACTIONS[action_idx]
        
        if save_movie:
            movie_frames.append(np.copy(ENV.get_current_screen()))
        
        state, reward, done = ENV.take_action(action, clip_rewards=False)
        total_reward += reward
    
    if save_movie:
        imageio.mimsave(os.path.join(RESULTS_FOLDER, movie_name + '.gif'), movie_frames)
    
    return total_reward


# In[12]:

models = get_model_list()
rewards = []
iterations = []

for i, (iteration, model_filepath) in enumerate(models):
    load_model(model_filepath)
    play_rewards = []
    
    for play in xrange(PER_MODEL_PLAYS):
    
        if play==0:
            save_movie = True
            movie_name = str(iteration)
        else:
            save_movie = False
            movie_name = None
            
        game_reward = play_game(save_movie, movie_name)
        play_rewards.append(game_reward)
        print "Iteration: %d, Play: %d, Reward: %f" % (iteration, play, game_reward)
    
    rewards.append(play_rewards)
    iterations.append(iteration)
    
    np.save(os.path.join(RESULTS_FOLDER, 'rewards'), np.array(rewards))
    np.save(os.path.join(RESULTS_FOLDER, 'iterations'), np.array(iterations))


# In[ ]:



