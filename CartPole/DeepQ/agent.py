import gym
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from collections import deque
from model import DQN
from buffer import ReplayBuffer


class Agent:
    """
    Class representing a learning agent acting in an environment.
    """
    
    def __init__(self, buffer_size, batch_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay, lr,
        game="CartPole-v1", mean_bound=5, reward_bound=495.0, sync_model=1000, save_model=10):
        """
        Constructor of the agent class.
            - game="CartPole-v1" : Name of the game environment
            - mean_bound=5 : Number of last acquired rewards considered for mean reward
            - reward_bound=495.0 : Reward acquired for completing an episode properly
            - sync_model=1000 : Interval for synchronizing model and target model
            - save_model=10 : Interval for saving model

            - buffer_size : Replay buffer size of the DQN model
            - batch_size : Batch size of the DQN model
            - alpha : Learning rate for Q-Learning
            - gamma : Discount factor for Q-Learning
            - epsilon : Threshold for taking a random action
            - epsilon_min : Minimal value allowed for epsilon
            - epsilon_decay : Decay rate for epsilon
            - lr : Learning rate for the DQN model
        """
        
        # Environment variables
        self.game = game
        self.env = gym.make(self.game)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.mean_bound = mean_bound
        self.reward_bound = reward_bound
        
        #DQN variables
        self.lr = lr
        self.model = DQN(self.num_states, self.num_actions, self.lr)
        self.target_model = DQN(self.num_states, self.num_actions, self.lr)
        self.target_model.update(self.model)
        self.sync_model = sync_model
        self.save_model = save_model

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/dqn.h5")
        self.path_plot = os.path.join(dirname, "plots/dqn.png")

        # Load model, if it already exists
        try:
            self.model.load(self.path_model)
            self.target_model.update(self.model)
        except:
            print("Model does not exist! Create new model...")


    def reduce_epsilon(self):
        """
        Reduces the parameter epsilon up to a given minimal value where the speed of decay is controlled by some given parameter.
        """

        epsilon = self.epsilon * self.epsilon_decay

        if epsilon >= self.epsilon_min:
            self.epsilon = epsilon
        else:
            self.epsilon = self.epsilon_min


    def get_action(self, state):
        """
        Returns an action for a given state, based on the current policy.
            - state : Current state of the agent
        """

        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state))

        return action

    
    def train(self, num_episodes, report_interval):
        """
        Trains the DQN model for a given number of episodes. Outputting report information is controlled by a given time interval.
            - num_episodes : Number of episodes to train
            - report_interval : Interval for outputting report information of training
        """

        step = 0
        total_rewards = []

        for episode in range(1, num_episodes + 1):
            if episode % self.save_model == 0:
                self.model.save(self.path_model)
            
            state = self.env.reset()
            state = state.reshape((1, self.num_states))
            total_reward = 0.0

            while True:
                step += 1
                
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape((1, self.num_states))

                # Penalize agent if pole could not be balanced until end of episode
                if done and reward < 499.0: reward = -100.0

                self.buffer.remember(state, action, reward, next_state, done)
                self.replay()
                self.reduce_epsilon()

                state = next_state
                total_reward += reward

                if step % self.sync_model == 0:
                    self.target_model.update(self.model)

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-self.mean_bound:])

                    if episode % report_interval == 0:
                        print(
                            f"Episode: {episode}/{num_episodes}"
                            f"\tStep: {step}"
                            f"\tMemory Size: {len(self.memory)}"
                            f"\tEpsilon: {self.epsilon : .3f}"
                            f"\tReward: {total_reward}"
                            f"\tLast 5 Mean: {mean_reward : .2f}")

                        self.plot_rewards(total_rewards)

                    if mean_reward > self.reward_bound:
                        self.model.save(self.path_model)
                        return

                    break

        self.model.save(self.path_model)

    
    def replay(self):
        """
        Samples training data from the replay buffer and fits the DQN model.
        """

        sample_size, states, actions, rewards, next_states, dones = self.memory.sample()

        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(sample_size):
            action = actions[i]
            done = dones[i]

            if done:
                q_target = rewards[i]
            else:
                q_target = rewards[i] + self.gamma * np.max(next_q_values[i])

            q_values[i][action] = (1 - self.alpha) * q_values[i][action] + self.alpha * q_target

        self.model.fit(states, q_values)


    def play(self, num_episodes):
        """
        Renders the trained agent for a given number of episodes.
            - num_episodes : Number of episodes to render
        """

        self.epsilon = self.epsilon_min

        for episode in range(1, num_episodes + 1):
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
                        f"Episode: {episode}/{num_episodes}"
                        f"\tTotal Reward: {total_reward : .2f}")

                    break


    def plot_rewards(self, total_rewards):
        """
        Plots the rewards the agent has acquired during training.
            - total_rewards : Rewards the agent has gained per episode
        """

        x = range(len(total_rewards))
        y = total_rewards

        slope, intercept, _, _, _ = linregress(x, y)
        
        plt.plot(x, y, linewidth=0.8)
        plt.plot(x, slope * x + intercept, color="red", linestyle="-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN-Learning")
        plt.savefig(self.path_plot)


if __name__ == "__main__":

    # Choose whether to play or to train
    PLAY = True
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5

    # Hyperparameters
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 128
    ALPHA = 0.2
    GAMMA = 0.95
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    LEARNING_RATE = 0.001

    agent = Agent(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        lr=LEARNING_RATE)
    
    if not PLAY:
        agent.train(num_episodes=EPISODES_TRAIN, report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)