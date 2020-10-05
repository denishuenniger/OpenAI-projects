import numpy as np
import pickle

from collections import defaultdict


class Q:

    def __init__(self, num_actions, alpha, gamma):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.table = defaultdict(lambda : [np.random.random() for _ in range(self.num_actions)])


    def fit(self, state, action, reward, next_state, done):
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.table[next_state])
        
        q = self.table[state][action]
        q = (1 - self.alpha) * q + self.alpha * q_target

        self.table[state][action] = q


    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(dict(self.table), file)

    
    def load(self, path):
        try:
            with open(path, "rb") as file:
                self.table = defaultdict(lambda : [0.0 for _ in range(self.num_actions)], pickle.load(file))
        except:
            print("Model does not exist! Create new model...")