import numpy as np
import random

from collections import deque


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.buffer_size)


    def __len__(self):
        return len(self.memory)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        sample_size = min(len(self.memory), self.batch_size)
        batch = random.sample(self.memory, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        return states, actions, rewards, next_states, dones