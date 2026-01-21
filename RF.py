from pickletools import optimize
from typing import Any


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)



class Memory_buffer():
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.buffer_rewards = deque(maxlen=max_size)
        self.buffer_logp = deque(maxlen=max_size)

    def push(self, item):
        self.buffer_rewards.append(item[0])
        self.buffer_logp.append(item[1])

    def clean(self):
        self.buffer_rewards.clear()
        self.buffer_logp.clear()


def get_action(policy, x):
    y = policy.forward(x)
    m = Categorical(y)
    action = m.sample()
    action_prob = m.log_prob(action)

    return action, action_prob



def update(policy, mem_buffer, **train_args):
    Gt = []
    R = 0
    loss = 0

    eps = train_args.eps
    gamma = train_args.gamma
    optim = train_args.optim
    lr = train_args.lr


    ## rewards list is [r_0, r_1, r_2, ...]
    rewards_list = list(mem_buffer.buffer_rewards[::-1])
    for r in rewards_list:
        R = r + R * gamma
        Gt.append(R)
    Gt = torch.tensor(Gt[::-1], dtype=torch.float32)
    Gt_nor = (Gt - Gt.mean()) / (Gt.std() + eps)

    for log_p, R in zip(mem_buffer.buffer_logp, Gt_nor):
        loss.append(-log_p * R)
    
    policy_loss = torch.cat(loss).sum()

    optimizer = optim.Adam(policy.parameters(), lr)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    mem_buffer.clean()


episode_num = 
state_size = 
action_size = 
hidden_state_size = 

stop_step = 

train_args = {
    'eps': 1e-5,
    'gamma': 0.99,
    'optim': optim.Adam,
    'lr': 0.001,
}

policy = Policy(state_size, action_size, hidden_state_size)
optim = 

for ii in range(episode_num):
    set_env()
    state = env.set()
    buffer = Memory_buffer()
    for ii in range(stop_step):
        action, action_log_prob = get_action(policy, state)
        reward, next_state, done = env.step(action, ii)
        buffer.push([reward, action_log_prob])
        
        update(policy, buffer, **train_args)
           
        state = next_state

        if done:
            break



    
    
    

