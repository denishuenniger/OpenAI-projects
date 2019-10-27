from collections import deque


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)


    def __len__(self):
        return len(self.buffer)


    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))