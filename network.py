"""
Networks
"""

import torch
import torch.nn as nn


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
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(ActorLow, self).__init__()
        self.fc = nn.Sequential(
            # (state, goal) -> action
            nn.Linear(state_dim + goal_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state, goal):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        if not isinstance(goal, torch.Tensor): state = torch.Tensor(goal)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        if len(goal.shape) < 2: goal = goal[None, :]
        # forward propagate
        obs = torch.cat([state, goal], 1)
        logits = self.fc(obs)
        action = logits * self.max_action
        return action


class CriticLow(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(CriticLow, self).__init__()
        # double Q networks
        self.fc1 = nn.Sequential(
            # (state, goal, action) -> q_l1
            nn.Linear(state_dim + goal_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.fc2 = nn.Sequential(
            # (state, goal, action) -> q_l2
            nn.Linear(state_dim + goal_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, goal, action):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        if not isinstance(goal, torch.Tensor): state = torch.Tensor(goal)
        if not isinstance(action, torch.Tensor): state = torch.Tensor(action)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        if len(goal.shape) < 2: goal = goal[None, :]
        if len(action.shape) < 2: action = action[None, :]
        # forward propagate
        obs_action = torch.cat([state, goal, action], 1)
        q1 = self.fc1(obs_action)
        q2 = self.fc2(obs_action)
        return q1, q2

    def q1(self, state, goal, action):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        if not isinstance(goal, torch.Tensor): state = torch.Tensor(goal)
        if not isinstance(action, torch.Tensor): state = torch.Tensor(action)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        if len(goal.shape) < 2: goal = goal[None, :]
        if len(action.shape) < 2: action = action[None, :]
        obs_action = torch.cat([state, goal, action], 1)
        # forward propagate
        q1 = self.fc1(obs_action)
        return q1


class ActorHigh(nn.Module):
    def __init__(self, state_dim, goal_dim, max_goal, device):
        super(ActorHigh, self).__init__()
        self.fc = nn.Sequential(
            # (state, goal) -> goal'
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, goal_dim),
            nn.Tanh()
        )
        self.max_goal = torch.Tensor(max_goal).to(device)

    def forward(self, state):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        # forward propagate
        logits = self.fc(state)
        next_goal = logits * self.max_goal
        return next_goal


class CriticHigh(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super(CriticHigh, self).__init__()
        self.fc1 = nn.Sequential(
            # (state, goal') -> q_h1
            nn.Linear(state_dim + goal_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.fc2 = nn.Sequential(
            # (state, goal, goal') -> q_h2
            nn.Linear(state_dim + goal_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, next_goal):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        obs_action = torch.cat([state, next_goal], 1)
        # forward propagate
        q1 = self.fc1(obs_action)
        q2 = self.fc2(obs_action)
        return q1, q2

    def q1(self, state, next_goal):
        # force to reformat input data
        if not isinstance(state, torch.Tensor): state = torch.Tensor(state)
        # reformat input as batch data
        if len(state.shape) < 2: state = state[None, :]
        obs_action = torch.cat([state, next_goal], 1)
        # forward propagate
        q1 = self.fc1(obs_action)
        return q1
