"""
Networks
"""

import torch
import torch.nn as nn
from torch.nn import functional


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        state = torch.Tensor(state)
        logits = self.fc(state)
        action = logits * self.max_action
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q1 = self.fc1(state_action)
        q2 = self.fc2(state_action)
        return q1, q2

    def q1(self, state, action):
        state_action = torch.cat([state, action], 1)
        q1 = self.fc1(state_action)
        return q1
