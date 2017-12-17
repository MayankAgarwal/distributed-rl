import numpy as np
from preprocess import Preprocess
import torch
import random
from ale_python_interface import ALEInterface

class Env(object):
    
    def __init__(self, rom_path, rescale_size, noop_range, state_tensor_type, state_frames=1, action_repeat=0):
        
        self._rompath = rom_path
        self._noop_range = noop_range
        self._state_tensor_type = state_tensor_type
        self._state_frames = state_frames
        self._action_repeat = action_repeat
        self._NOOP_ACTION = 0
        
        self.__init_ale()
        self.action_set = self._ale.getMinimalActionSet()
        self._w, self._h = self._ale.getScreenDims()
        
        self._frames_hist = []
        self._preprocessor = Preprocess(rescale_size)
        
        self.reset_game()
        
    def __init_ale(self):
        
        self._ale = ALEInterface()
        self._ale.setBool('display_screen', False)
        self._ale.loadROM(self._rompath)
        
    def __init_frames_hist(self):
        temp = self.get_current_screen()
        self._frames_hist = [temp]*(self._state_frames + 1)
        
    def __add_frame(self, frame):
        self._frames_hist.append(frame)
        self._frames_hist = self._frames_hist[-1*(self._state_frames+1):]
   
    def get_current_screen(self):
        temp = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        self._ale.getScreenRGB(screen_data=temp)
        return temp

    def get_state(self):
        
        if self.__isdone():
            return None
        
        preprocessed = self._preprocessor.process_images(self._frames_hist)
        state_tensor = torch.from_numpy(preprocessed).type(self._state_tensor_type)
        
        if len(state_tensor.size()) == 3:  # 3d tensor
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor
    
    def __isdone(self):
        return self._ale.lives() <= 0
    
    def take_action(self, action, clip_rewards=True):

        i = 0
        total_reward = 0.0

        while not self.__isdone() and i <= self._action_repeat:
            reward = self._ale.act(action)
            if clip_rewards: 
                reward = min(1, max(reward, -1))
            total_reward += reward
            i += 1
        
        screen_img = self.get_current_screen()
        self.__add_frame(screen_img)
        
        return self.get_state(), total_reward, self.__isdone()
    
    def reset_game(self):
        self._ale.reset_game()
        self.__take_noop_actions()
        self.__init_frames_hist()
        return self.__isdone()
        
    def __take_noop_actions(self):
        
        noop_actions = random.randint(*(self._noop_range))
        while noop_actions > 0:
            self._ale.act(self._NOOP_ACTION)
            noop_actions -= 1