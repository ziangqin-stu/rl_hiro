""""
Off-Policy Method (TD3) Experience Memory Buffer
"""


import torch


class ExperienceMemory:
    """
    DevNotes:
        - data format, data shape
        - gpu/cup
        - different policies'/envs' requirements
    """
    def __init__(self, capacity):
        # initialize
        self.capacity = capacity
        self.offset = 0
        # create experience buffer
        self.state = torch.zeros(capacity).cuda()
        pass

    def reset(self):
        self.__init__(self.capacity)

    def append(self, state_batch):
        # insert range
        batch_size = len(state_batch)
        start, end = self.offset, self.offset + batch_size
        # insert experience batch
        self.state[start, end] = state_batch[start: end]
        pass

    def sample(self):
        pass

    def compute(self):
        pass



