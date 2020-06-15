""""
Off-Policy Method (TD3) Experience Memory Buffer
"""


import torch
import numpy as np


class ExperienceBuffer:
    """
    DevNotes:
        - data format, data shape
        - gpu/cup
        - different policies'/envs' requirements
    """
    def __init__(self, capacity, state_dim, action_dim):
        # initialize
        self.capacity = capacity
        self.offset = 0
        # create experience buffer
        self.state = torch.zeros(capacity, state_dim)
        self.next_state = torch.zeros(capacity, state_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.reward = torch.zeros(capacity, 1)
        self.done = torch.zeros(capacity, 1)
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def compute(self):
        pass



