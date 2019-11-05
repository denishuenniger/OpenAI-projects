import random
import numpy as np

from collections import deque


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)


    def __len__(self):
        return len(self.buffer)


    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self):
        sample_size = min(self.__len__(), self.batch_size)
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        return sample_size, states, actions, rewards, next_states, dones