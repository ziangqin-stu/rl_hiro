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
    def __init__(self, capacity, state_dim, action_dim):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

    def reset(self):
        self.__init__(self.capacity)

    def add(self, state, action, next_state, reward, done):
        ind = self.offset
        self.state[ind] = torch.Tensor(state)
        self.action[ind] = torch.Tensor(action)
        self.next_state[ind] = torch.Tensor(next_state)
        self.reward[ind] = torch.Tensor([reward])[:, None]
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
    def __init__(self, capacity, state_dim, action_dim):
        # initialize
        self.capacity = capacity
        self.offset = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create experience buffer
        self.state = torch.zeros(capacity, state_dim)
        self.goal = torch.zeros(capacity, state_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.next_state = torch.zeros(capacity, state_dim)
        self.next_goal = torch.zeros(capacity, state_dim)
        self.reward = torch.zeros(capacity, 1)
        self.done = torch.zeros(capacity, 1)
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

    def reset(self):
        self.__init__(self.capacity, self.state_dim, self.action_dim)

    def add(self, state, goal, action, next_state, next_goal, reward, done):
        ind = self.offset
        self.state[ind] = torch.Tensor(state)
        self.goal[ind] = torch.Tensor(goal)
        self.action[ind] = torch.Tensor(action)
        self.next_state[ind] = torch.Tensor(next_state)
        self.next_goal[ind] = torch.Tensor(next_goal)
        self.reward[ind] = torch.Tensor(reward)
        self.done[ind] = torch.Tensor(done)
        self.offset += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self. offset + 1, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


class ExperienceBufferHigh:
    """
    DevNotes:
        - hold cpu data in memory, transform when output
    """
    def __init__(self, capacity, c, state_dim, action_dim):
        # initialize
        self.capacity = capacity
        self.offset = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create experience buffer
        self.state = torch.zeros(capacity, c, state_dim)
        self.goal = torch.zeros(capacity, c, state_dim)
        self.action = torch.zeros(capacity, c, action_dim)
        self.next_state = torch.zeros(capacity, c, state_dim)
        self.next_goal = torch.zeros(capacity, c, state_dim)
        self.reward = torch.zeros(capacity, c, 1)
        self.done = torch.zeros(capacity, c, 1)
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

    def reset(self):
        self.__init__(self.capacity, self.state_dim)

    def add(self, state, goal, action, next_state, next_goal, reward, done):
        ind = self.offset
        self.state[ind] = torch.stack(state)
        self.goal[ind] = torch.stack(goal)
        self.action[ind] = torch.stack(action)
        self.next_state[ind] = torch.stack(next_state)
        self.next_goal[ind] = torch.stack(next_goal)
        self.reward[ind] = torch.stack(reward)
        self.done[ind] = torch.stack(done)
        self.offset += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self. offset + 1, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


