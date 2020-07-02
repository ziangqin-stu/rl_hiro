""""
Off-Policy Method (TD3) Experience Memory Buffer
"""


import torch
import numpy as np


class ExperienceBufferTD3:
    """
    DevNotes:
        - assume state, action is not one-dimension
        - hold cpu data in memory, transform when output
    """
    def __init__(self, capacity, state_dim, action_dim, use_cuda):
        # initialize
        self.capacity = capacity
        self.offset = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create experience buffer
        self.state = torch.zeros(capacity, state_dim)
        self.next_state = torch.zeros(capacity, state_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.reward = torch.zeros(capacity, 1)
        self.done = torch.zeros(capacity, 1)
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
        # self.device = "cpu"

    def reset(self):
        self.__init__(self.capacity)

    def add(self, state, action, reward, next_state, done):
        ind = self.offset
        self.state[ind] = torch.Tensor(state)
        self.action[ind] = torch.Tensor(action)
        self.reward[ind] = torch.Tensor([reward])[:, None]
        self.next_state[ind] = torch.Tensor(next_state)
        self.done[ind] = torch.Tensor([done])[:, None]
        self.offset += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self. offset + 1, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


class ExperienceBufferLow:
    """
    DevNotes:
        - assume state, action is not one-dimension
        - hold cpu data in memory, transform when output
    """
    def __init__(self, capacity, state_dim, goal_dim, action_dim, use_cuda):
        # initialize
        self.capacity = capacity
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.offset = 0
        # create experience buffer
        self.state = torch.zeros(capacity, state_dim)
        self.goal = torch.zeros(capacity, goal_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.reward = torch.zeros(capacity, 1)
        self.next_state = torch.zeros(capacity, state_dim)
        self.next_goal = torch.zeros(capacity, goal_dim)
        self.done = torch.zeros(capacity, 1)
        # probe device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
        # self.device = "cpu"

    def add(self, state, goal, action, reward, next_state, next_goal, done):
        # add step experience (off-policy single steps)
        ind = self.offset
        self.state[ind] = state.cpu()
        self.goal[ind] = goal.cpu()
        self.action[ind] = action.cpu()
        self.reward[ind] = reward.cpu()
        self.next_state[ind] = next_state.cpu()
        self.next_goal[ind] = next_goal.cpu()
        self.done[ind] = done.cpu()
        self.offset += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self. offset + 1, size=batch_size)
        return (
            self.state[ind].to(self.device),
            self.goal[ind].to(self.device),
            self.action[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.next_goal[ind].to(self.device),
            self.done[ind].to(self.device)
        )


class ExperienceBufferHigh:
    """
    DevNotes:
        - hold cpu data in memory, transform when output
    """
    def __init__(self, capacity, state_dim, goal_dim, use_cuda):
        # initialize
        self.capacity = capacity
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.offset = 0
        # create experience buffer
        self.state_start = torch.zeros(capacity, state_dim)
        self.goal_start = torch.zeros(capacity, goal_dim)
        self.reward = torch.zeros(capacity, 1)
        self.state_end = torch.zeros(capacity, state_dim)
        self.done = torch.zeros(capacity, 1)
        # probe device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
        # self.device = "cpu"

    def add(self, state_start, goal_start, reward, state_end, done):
        # add step experience (off-policy single steps)
        ind = self.offset
        self.state_start[ind] = state_start.cpu()
        self.goal_start[ind] = goal_start.cpu()
        self.reward[ind] = reward.cpu()
        self.state_end[ind] = state_end.cpu()
        self.done[ind] = done.cpu()
        self.offset += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self. offset + 1, size=batch_size)
        return (
            self.state_start[ind].to(self.device),
            self.goal_start[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.state_end[ind].to(self.device),
            self.done[ind].to(self.device)
        )


