import random

class ReplayMemory(object):
    
    def __init__(self, capacity, data_obj):
        
        self.capacity = capacity
        self.data_obj = data_obj
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = self.data_obj(*args)
        self.position = (self.position + 1)%self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)