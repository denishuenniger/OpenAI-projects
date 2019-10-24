import gym
import os
import random
import numpy as np

from collections import deque
from model import DQN


class Agent:

    def __init__(self, env, replay_buffer_size, train_start,
                    gamma, alpha, batch_size, learning_rate):
        # Environment variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.replay_buffer_size = replay_buffer_size
        self.train_start = train_start
        self.memory = deque(maxlen=self.replay_buffer_size)
        self.gamma = gamma
        self.alpha = alpha
        
        #DQN variables
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = DQN(self.num_states, self.num_actions, self.learning_rate)
        self.target_model = DQN(self.num_states, self.num_actions, self.learning_rate)
        self.target_model.update(self.model)

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/dqn.h5")
        self.path_plot = os.path.join(dirname, "plots/dqn.png")


    def get_action(self, state):
        policy = self.model.predict(state)[0]
        action = np.random.choice(self.num_actions, p=policy)

        return action


    def train(self, num_episodes):
        try:
            self.model.load(self.path_model)
        except:
            print("Model does not exist! Create new model...")
        
        total_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            state = state.reshape((1, self.num_states))
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape((1, self.num_states))

                if done and reward != 500.0: reward = -100.0

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    mean_total_rewards = np.mean(total_rewards[-10:])

                    print(f"Episode: {episode + 1}/{num_episodes} \tTotal Reward: {total_reward} \tMean Total Rewards: {mean_total_rewards}")

                    if mean_total_rewards > 495.0:
                        self.model.save(self.path_model)
                        return total_rewards

                    break

        self.model.save(self.path_model)
        return total_rewards


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self):
        if len(self.memory) < self.train_start: return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            action = actions[i]
            done = dones[i]

            if done:
                q_values[i][action] = rewards[i]
            else:
                q_values[i][action] = rewards[i] + self.gamma * np.max(next_q_values[i])

        self.model.fit(states, q_values)


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
                    print(f"Episode: {episode + 1}/{num_episodes} \tTotal Reward: {total_reward}")
                    break


    def plot_rewards(self, total_rewards):
        plt.plot(range(len(total_rewards)), total_rewards, linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN - Learning")
        plt.savefig(self.path_plot)



if __name__ == "__main__":

    # Hyperparameters
    REPLAY_BUFFER_SIZE = 500000
    TRAIN_START = 1000
    GAMMA = 0.95
    ALPHA = 0.2
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    PLAY = False
    EPISODES_TRAIN = 10000
    EPISODES_PLAY = 5


    env = gym.make("CartPole-v1")
    agent = Agent(env, 
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                train_start=TRAIN_START,
                gamma=GAMMA,
                alpha=ALPHA,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE)
    
    if not PLAY:
        total_rewards = agent.train(num_episodes=EPISODES_TRAIN)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)

