import gym
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from model import DQN
from buffer import ReplayBuffer


class Agent:

    def __init__(
        self, env, buffer_size, batch_size,
        alpha, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate):
        # Environment variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        # Agent variables
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        #DQN variables
        self.learning_rate = learning_rate
        self.model = DQN(self.num_states, self.num_actions, self.learning_rate)
        self.target_model = DQN(self.num_states, self.num_actions, self.learning_rate)
        self.target_model.update(self.model)

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/dqn.h5")
        self.path_plot = os.path.join(dirname, "plots/dqn.png")


    def reduce_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay

        if epsilon >= self.epsilon_min:
            self.epsilon = epsilon
        else:
            self.epsilon = self.epsilon_min


    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.model.predict(state)[0]

        return action


    def replay(self):
        sample_size = min(len(self.buffer.memory), self.batch_size)
        states, actions, rewards, next_states, dones = self.buffer.sample()

        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(sample_size):
            if dones[i]:
                q_target = rewards[i]
            else:
                q_target = rewards[i] + self.gamma * np.max(next_q_values[i])

            q_values[i] = (1 - self.alpha) * q_values[i] + self.alpha * q_target

        self.model.fit(states, q_values)

    
    def train(self, num_episodes, report_interval):
        try:
            self.model.load(self.path_model)
        except:
            print("Model does not exist! Create new model...")
        
        total_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            state = state.reshape(1, self.num_states)
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.num_states)

                self.buffer.remember(state, action, reward, next_state, done)
                self.replay()
                self.reduce_epsilon()

                state = next_state
                total_reward += reward

                if done:
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards)

                    if (episode + 1) % report_interval == 0:
                        print(
                            f"Episode: {episode + 1}/{num_episodes}"
                            f"\tReward: {total_reward : .2f}"
                            f"\tMean Reward: {mean_reward : .2f}")

                    self.target_model.update(self.model)
                    break

        self.model.save(self.path_model)
        return total_rewards


    def play(self, num_episodes):
        self.model.load(self.path_model)

        for episode in range(num_episodes):
            state = self.env.reset()
            state = state.reshape((1, self.num_states))
            total_reward = 0.0

            while True:
                self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape((1, self.num_states))
                state = next_state
                total_reward += reward

                if done:
                    print(
                        f"Episode: {episode + 1}/{num_episodes}"
                        f"\tTotal Reward: {total_reward}")
                    break


    def plot_rewards(self, total_rewards):
        x = range(len(total_rewards))
        y = total_rewards

        slope, intercept, _, _, _ = linregress(x, y)
        
        plt.plot(x, y, linewidth=0.8)
        plt.plot(x, slope * x + intercept, color="r", linestyle="-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":

    # Hyperparameters
    BUFFER_SIZE = 1000000
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    PLAY = False
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 10000
    EPISODES_PLAY = 5


    env = gym.make("BipedalWalker-v2")
    agent = Agent(
        env=env, 
        buffer_size=BUFFER_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE)
    
    if not PLAY:
        total_rewards = agent.train(
            num_episodes=EPISODES_TRAIN,
            report_interval=REPORT_INTERVAL)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)