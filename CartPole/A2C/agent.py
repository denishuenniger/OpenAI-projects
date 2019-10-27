import gym
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from model import DNN
from buffer import ReplayBuffer


class Agent:

    def __init__(
        self, env, alpha, gamma, buffer_size,
        epsilon, epsilon_min, epsilon_decay,
        batch_size, lr_actor, lr_critic):
        # Environment variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.buffer = ReplayBuffer(buffer_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # DQN variables
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.model = DNN(self.num_states, self.num_actions, self.lr_actor, self.lr_critic)

        # File paths
        directory = os.path.dirname(__file__)
        self.path_actor = os.path.join(directory, "models/actor.h5")
        self.path_critic = os.path.join(directory, "models/critic.h5")
        self.path_plot = os.path.join(directory, "plots/dnn.png")


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
            action = np.argmax(self.model.predict_actor(state))

        return action


    def replay(self):
        sample_size = min(len(self.buffer), self.batch_size)
        minibatch = random.sample(self.buffer.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        values = np.zeros((sample_size, 1))
        advantages = np.zeros((sample_size, self.num_actions))

        q_values = self.model.predict_critic(states)
        next_q_values = self.model.predict_critic(next_states)

        for i in range(sample_size):
            action = actions[i]
            done = dones[i]

            if done:
                advantages[i][action] = rewards[i] - q_values[i]
                values[i] = rewards[i]
            else:
                advantages[i][action] = (rewards[i] + self.gamma * next_q_values[i]) - q_values[i]
                values[i] = rewards[i] + self.gamma * next_q_values[i]

        self.model.fit_actor(states, advantages)
        self.model.fit_critic(states, values)
    
    
    def train(self, num_episodes, report_interval):
        try:
            self.model.load_actor(self.path_actor)
            self.model.load_critic(self.path_critic)
        except:
            print(f"Model does not exist! Create new model...")

        total_rewards = []

        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = state.reshape(1, self.num_states)

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.num_states)

                if done and reward != 500.0: reward = -100.0

                self.buffer.remember(state, action, reward, next_state, done)
                self.replay()
                self.reduce_epsilon()

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

                    if mean_reward > 495.0:
                        self.model.save_actor(self.path_actor)
                        self.model.save_critic(self.path_critic)

                        return total_rewards

                    break

        self.model.save_actor(self.path_actor)
        self.model.save_critic(self.path_critic)

        return total_rewards


    def play(self, num_episodes):
        self.model.load_actor(self.path_actor)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state = state.reshape(1, self.num_states)

            while True:
                self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                state = state.reshape(1, self.num_states)

                if done:
                    print(
                        f"Episode: {episode + 1}"
                        f"\t Reward: {reward : .2f}")
                    break


    def plot_rewards(self, total_rewards):
        x = range(len(total_rewards))
        y = total_rewards

        slope, intercept, _, _, _ = linregress(x, y)
        
        plt.plot(x, y, linewidth=0.8)
        plt.plot(x, slope * x + intercept, color="r", linestyle="-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("A2C-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":

    # Hyperparameters
    BUFFER_SIZE = 1000000
    ALPHA = 0.2
    GAMMA = 0.98
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99
    BATCH_SIZE = 128
    LR_ACTOR = 0.0001
    LR_CRITIC = 0.001

    PLAY = False
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5

    env = gym.make("CartPole-v1")
    agent = Agent(
        env=env,
        buffer_size=BUFFER_SIZE,
        alpha=ALPHA,
        gamma = GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC)
    
    if not PLAY:
        total_rewards = agent.train(
            num_episodes=EPISODES_TRAIN,
            report_interval=REPORT_INTERVAL)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)