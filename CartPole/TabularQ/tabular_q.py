import matplotlib.pyplot as plt
import numpy as np
import gym
import pickle
import time
from collections import defaultdict
from scipy.stats import linregress


ALPHA = 0.75
FPS = 60
GAMMA = 0.9
INTERVAL = 10000
NUM_EPISODES = 1000000
RENDER = False
THRESHOLD = 0.01


def print_result(rewards, time):
    print(
        f"\n{'-' * 50}\n"
        f'Average Reward: {np.mean(rewards) : .2f} (Episode {NUM_EPISODES})\n'
        f'Simulation Duration: {time : .2f} min\n'
        f"{'-' * 50}\n"
    )


def plot_learning_curve(rewards):
    x = np.arange(1, NUM_EPISODES + 1)
    y = rewards

    slope, intercept, _, _, _ = linregress(x, y)

    plt.figure(figsize=(15, 5))
    plt.plot(x, y, linewidth=0.5)
    plt.plot(x, slope * x + intercept)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Simple Q-Learning - Learning Curve')
    plt.text(9000,max(y),slope)
    plt.show()


class CartPole:

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.file = 'Q.pickle'
        self.Q = defaultdict(lambda : [np.random.random(), np.random.random()])


    def load(self):
        try:
            with open(self.file, 'rb') as file:
                self.Q =  defaultdict(lambda : [0, 0], pickle.load(file))
        except:
            print(f'File does not exist! Default file is loaded.')


    def save(self):
        with open(self.file, 'wb') as file:
            pickle.dump(dict(self.Q), file)


    def simulate(self):
        self.env.seed(0)
        np.random.seed(0)

        rewards = []
        start = time.time()

        for episode in range(1, NUM_EPISODES + 1):
            state = self.convert_state(self.env.reset())
            episode_reward = 0

            render = 0

            while True:

                state, reward, done = self.train(state)
                episode_reward += reward
                if RENDER:
                    if render % 10 == 0:
                        self.env.render()
                    render += 1
                if done:
                    rewards.append(episode_reward)

                    if episode % INTERVAL == 0:
                        self.print_reward(episode_reward, episode)

                    break

        self.env.close()

        stop = time.time()
        elapsed = (stop - start) / 60

        return rewards, elapsed


    def train(self, state):
        action = self.Q[state].index(max(self.Q[state]))
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.convert_state(next_state)

        Q_target = reward + GAMMA * max(self.Q[next_state])
        self.Q[state][action] = (1-ALPHA)*self.Q[state][action] + ALPHA*Q_target

        return next_state, reward, done


    @staticmethod
    def convert_state(state):
        return '_'.join(map(str, [round(value, 1) for value in state]))


    @staticmethod
    def print_reward(reward, episode):
        print(f'Reward: {reward: .2f} (Episode {episode})')



if __name__ == '__main__':

    model = CartPole()
    model.load()
    rewards, time = model.simulate()
    model.save()

    print_result(rewards, time)
    plot_learning_curve(rewards)