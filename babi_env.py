'''
OpenAI Gym Q-learning agent for simple bAbI tasks
Author: Ruaridh George
Date: 07/2020
python gym-babi/gym_babi/envs/babi_env.py
'''

from gym import Env, spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding, colorize
from babi_loader import BabiDataset, pad_collate

# new task, new environment, new vocab index
# the agent needs to know difference between returning co

class BabiEnv(Env):
    ''' 3-tuple ([[s1], [s2], [s3]], [question], answer)
                (context, question, answer)
    '''
    
    def __init__(self, task_id=1, seed=123, mode='train'):
        
        self.seed = np.random.seed(seed)
        self.i = 0
        self.num_correct = 0
        self.task_id = task_id
        self.last_action = None
        
        self.data = BabiDataset(task_id, mode)
        self.current_qa = self.data[self.i]
        self.vocab_size = len(self.data.QA.VOCAB)
        
        self.action_space = list(self.data.QA.VOCAB.values()) #  0 = no answer
        self.state_space = list(self.data.QA.VOCAB.values())  #  0 = start of QA
        
        
        self.state = self.current_qa[0][0][0]
        self.pos_idx = {'q': 0, 'sen': 0, 'word': 0}
        
    def __getitem__(self):
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return 'BabiEnv({0})'.format(task_id)
        
    def context_len(self):
        return len(self.current_qa[1])

    def get_reward(self, action):
        answer = self.current_qa[2]
        if action == answer:
            return 100
        elif action == 0:
            return 1
        elif action == answer and pos_idx['q'] == 1:
            return 200
        else:
            return -100
            
    def end_of_qa(self):
        self.i += 1
        self.i = self.i % (len(self.data) - 1)
        self.current_qa = self.data[self.i]
        self.pos_idx = {'q': 0, 'sen': 0, 'word': 0}
        
    def step(self, action):
        
        done = False
        reward = self.get_reward(action)
        target_act = self.current_qa[2] if self.pos_idx['q'] == 1 \
                      and self.pos_idx['word'] > 1 else 0
            
        if self.pos_idx['sen'] == len(self.current_qa[0]) - 1 \
        and self.state == 1:                                         # end of context
            new_state = self.current_qa[1][0]
            self.pos_idx = {'q': 1, 'sen': 0, 'word': 0}            # move to question
        
        elif self.pos_idx['q'] == 0 \
        and self.state == self.data.QA.VOCAB['<EOS>']:             # end of cntx sentence
            self.pos_idx['sen'] += 1                                # move to next sentence
            self.pos_idx['word'] = 0
            new_state = self.current_qa[0][self.pos_idx['sen']][0]  
        
        elif self.pos_idx['q'] == 1 \
        and self.pos_idx['word'] == len(self.current_qa[1]) - 1:    # end of question
            done = True
            self.end_of_qa()
            new_state = self.current_qa[0][0][0]
        
        else:                                                       # middle of sentence
            self.pos_idx['word'] += 1
            if self.pos_idx['q'] == 0:
                new_state = self.current_qa[self.pos_idx['q']] \
                            [self.pos_idx['sen']][self.pos_idx['word']]
            else:
                new_state = self.current_qa[self.pos_idx['q']][self.pos_idx['word']]
        
        self.state = new_state
        self.last_action = action
        
        return new_state, reward, done, {"target_act": target_act}
            

# return tensor            
    def reset(self):
        self.state = self.data[0][0][0][0]
        self.current_qa = self.data[self.i]
        return self.state
    
    def render(self, mode='human'):
        print('------------------------')
        print('State is in question: {0}'.format(bool(self.pos_idx['q'])))
        print('Sentence number:', colorize(str(self.pos_idx['sen']), 'yellow'))
        print('Current word state:', colorize(str(self.data.QA.IVOCAB[self.state]), 'cyan'))
        if self.last_action == self.current_qa[2]:
            print(colorize('Correct action! The answer is', 'green'), \
            self.data.QA.IVOCAB[self.last_action])
            if self.pos_idx['q']==1 and self.pos_idx['word']>1:
                self.num_correct += 1
        else:
            None
        print('Correct answers:', str(self.num_correct))
        print('Current QA:', colorize(str(self.i), 'magenta')) 
        print('------------------------')
        
      
    def close(self):
        print('Environment closed')
    
#- use assert for debugging functions
#- gamma is for balancing immediate and future reward
#- balanced accuracy accounts for imbalanced data