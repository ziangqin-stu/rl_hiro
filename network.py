"""
Networks
"""

import torch
import torch.nn as nn
from torch.nn import functional


class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorTD3, self).__init__()
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


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticTD3, self).__init__()
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


class ActorLow(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorLow, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state, goal):
        state, goal = torch.Tensor(state), torch.Tensor(goal)
        if len(state.shape) < 2: state = state[None, :]  # reformat as batch data
        if len(goal.shape) < 2: goal = goal[None, :]
        obs = torch.cat([state, goal], 1)
        logits = self.fc(obs)
        action = logits * self.max_action
        return action


class CriticLow(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticLow, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, goal, action):
        state, goal, action = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(action)
        obs_action = torch.cat([state, goal, action], 1)
        q1 = self.fc1(obs_action)
        q2 = self.fc2(obs_action)
        return q1, q2

    def q1(self, state, goal, action):
        state, goal, action = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(action)
        obs_action = torch.cat([state, goal, action], 1)
        q1 = self.fc1(obs_action)
        return q1


class ActorHigh(nn.Module):
    def __init__(self, c, state_dim, max_goal):
        super(ActorHigh, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c * state_dim * 2, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, state_dim),
            # nn.Tanh()
        )
        self.max_goal = max_goal

    def forward(self, state, goal):
        state, goal = torch.Tensor(state), torch.Tensor(goal)
        if len(state.shape) < 3: state = state[None, :]  # reformat as batch data
        if len(goal.shape) < 3: goal = goal[None, :]
        obs = torch.cat([state, goal], 2)
        if len(obs.shape) < 3: obs = obs[None, :]
        obs = obs.flatten(start_dim=1)
        logits = self.fc(obs)
        next_goal = logits * self.max_goal
        return next_goal


class CriticHigh(nn.Module):
    def __init__(self, c, state_dim):
        super(CriticHigh, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c * state_dim * 2 + state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(c * state_dim * 2 + state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, goal, next_goal):
        state, goal, next_goal = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(next_goal)
        obs_action = torch.cat([torch.cat([state, goal], 2).flatten(start_dim=1), next_goal], 1)
        q1 = self.fc1(obs_action)
        q2 = self.fc2(obs_action)
        return q1, q2

    def q1(self, state, goal, next_goal):
        state, goal, next_goal = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(next_goal)
        obs_action = torch.cat([torch.cat([state, goal], 2).flatten(start_dim=1), next_goal], 1)
        q1 = self.fc1(obs_action)
        return q1






