import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Class representing the experience replay buffer of the DQN model.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Constructor of the replay buffer class.
            - buffer_size : Buffer size of the DQN model
            - batch_size : Batch size of the DQN model
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        """
        Returns the length of the replay buffer.
        """

        return len(self.buffer)

    def remember(self, state, action, reward, next_state, done):
        """
        Adds a new element to the end of the replay buffer.
            - state : Current state of the agent
            - action : Action the agent decides to execute
            - reward : Reward the agent has gained for its chosen action
            - next_state : Next state the agent is located after executing its chosen action
            - done : Flag whether current episode is finished
        """

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Returns a sample of training data of the replay buffer.
        """

        sample_size = min(self.__len__(), self.batch_size)
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        return sample_size, states, actions, rewards, next_states, dones
