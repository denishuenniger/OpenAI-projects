import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from model import Q

class Agent:

    def __init__(
        self, env, gamma, alpha,
        epsilon, epsilon_min, epsilon_decay):
        # Environment variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.alpha = alpha
        self.gamma = gamma
        self.q = Q(self.num_actions, self.alpha, self.gamma)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/q.pickle")
        self.path_plot = os.path.join(dirname, "plots/q.png")


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
            action = np.argmax(self.q.table[state])

        return action


    def train(self, num_episodes, report_interval):
        try:
            self.q.load(self.path_model)
        except:
            print("Model does not exist! Create new model...")
        
        total_rewards = []  
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.get_state(state)
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.get_state(next_state)
                
                if done and reward != 500.0: reward = -100.0
                
                self.q.fit(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-5:])

                    if (episode + 1) % report_interval == 0:
                        print(
                            f"Episode: {episode + 1}/{num_episodes}"
                            f"\tReward: {total_reward : .2f}"
                            f"\tMean Reward: {mean_reward : .2f}")

                    if mean_reward >= 495.0:
                        self.q.save(self.path_model)
                        return total_rewards

                    break

        self.q.save(self.path_model)
        return total_rewards


    def play(self, num_episodes):
        self.q.load(self.path_model)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    print(
                        f"Episode: {episode + 1}"
                        f"\tReward: {total_reward : .2f}")
                    break


    def plot_rewards(self, total_rewards):
        x = range(len(total_rewards))
        y = total_rewards

        slope, intercept, _, _, _ = linregress(x, y)
        
        plt.plot(x, y, linewidth=0.8)
        plt.plot(x, slope * x + intercept, color="r", linestyle="-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Tabular Q-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":

    # Hyperparameters
    GAMMA = 0.95
    ALPHA = 0.2
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98

    PLAY = False
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5
    
    env = gym.make("CartPole-v1") 
    agent = Agent(
        env=env,
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY)

    if not PLAY:
        total_rewards = agent.train(
            num_episodes=EPISODES_TRAIN,
            report_interval=REPORT_INTERVAL)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)