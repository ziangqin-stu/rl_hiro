"""
Networks
"""

import torch
import torch.nn as nn


class Critic:
    def __init__(self, state_dim, action_dim):
        # build compute graph
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(action_dim + 300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        # initialize network

    def forward(self, state, action):
        state, action = state.flatten(), action.flatten()
        x = self.fc(torch.stack(state.flatten(), action.flatten()))
        return x


class Actor:
    def __init__(self, state_dim):
        # build compute graph
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Tanh()
        )
        # initialize network

    def forward(self, state):
        state, action = state.flatten(), action.flatten()
        x = self.fc(torch.stack(state.flatten(), action.flatten()))
        return x
