import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from model import Q

class Agent:

    def __init__(self, gamma, alpha, epsilon, epsilon_min, epsilon_decay,
        game="CartPole-v1", mean_bound=5, reward_bound=495.0, save_model=10):
        # Environment variables
        self.game = game
        self.env = gym.make(self.game)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.alpha = alpha
        self.gamma = gamma
        self.model = Q(self.num_actions, self.alpha, self.gamma)
        self.save_model = save_model
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.mean_bound = mean_bound
        self.reward_bound = reward_bound

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/q.pickle")
        self.path_plot = os.path.join(dirname, "plots/q.png")

        # Load model, if it already exists
        try:
            self.model.load(self.path_model)
        except:
            print("Model does not exist! Create new model...")


    def reduce_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay

        if epsilon >= self.epsilon_min:
            self.epsilon = epsilon
        else:
            self.epsilon = self.epsilon_min


    def get_state(self, state):
        return "_".join([str(np.round(x, 2)) for x in state])


    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.table[state])

        return action


    def train(self, num_episodes, report_interval):
        total_rewards = []  
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            state = self.get_state(state)
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.get_state(next_state)
                
                # Penalize agent if pole could not be balanced until end of episode
                if done and reward < 499.0: reward = -100.0
                
                self.model.fit(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-self.mean_bound:])

                    if episode % report_interval == 0:
                        print(
                            f"Episode: {episode}/{num_episodes}"
                            f"\tEpsilon: {self.epsilon : .3f}"
                            f"\tReward: {total_reward : .2f}"
                            f"\tMean Reward: {mean_reward : .2f}")

                        self.plot_rewards(total_rewards)

                    if mean_reward >= 495.0:
                        self.model.save(self.path_model)
                        return

                    break

        self.model.save(self.path_model)


    def play(self, num_episodes):
        self.epsilon = self.epsilon_min
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    print(
                        f"Episode: {episode}"
                        f"\tReward: {total_reward : .2f}")
                    break


    def plot_rewards(self, total_rewards):
        x = range(len(total_rewards))
        y = total_rewards

        slope, intercept, _, _, _ = linregress(x, y)
        
        plt.plot(x, y, linewidth=0.8)
        plt.plot(x, slope * x + intercept, color="red", linestyle="-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Tabular Q-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":

    PLAY = False
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5

    # Hyperparameters
    GAMMA = 0.95
    ALPHA = 0.2
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    
    agent = Agent(
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY)

    if not PLAY:
        agent.train(num_episodes=EPISODES_TRAIN, report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)