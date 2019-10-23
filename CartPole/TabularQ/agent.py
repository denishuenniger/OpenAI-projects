import gym
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


class Agent:

    def __init__(self, env, gamma, alpha, epsilon):
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.q = defaultdict(lambda : [0.0 for _ in range(self.num_actions)])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/q.pickle")
        self.path_plot = os.path.join(dirname, "plots/q.png")


    def convert_state(self, state):
        print(state)
        return '_'.join([str(np.round(x, 2)) for x in state])

    
    def get_action(self, state):
        state = self.convert_state(state)
        action = np.argmax(self.q[state])

        return action


    def get_value(self, state):
        state = self.convert_state(state)
        values = np.max(self.q[state])

        return value


    def get_random_action(self):
        action = self.env.action_space.sample()
        return action


    def update_q_values(self, state, action, reward, next_state):
        state = self.convert_state(state)
        next_state = self.convert_state(next_state)
        
        q_update = reward + self.gamma * self.get_value(next_state)
        q = self.q[state][action]
        q = (1 - self.alpha) * q + self.alpha * q_update

        self.q[state][action] = q


    def save_q_values(self):
        with open(self.path_model, "wb") as file:
            pickle.dumps(self.q)

    
    def load_q_values(self):
        try:
            with open(self.path_model, "rb") as file:
                self.q = pickle.load(file)
        except:
            print("Model does not exist! Create new model...")


    def train(self, num_episodes):
        total_rewards = []
        self.load_q_values()  
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if np.random.random() <= self.epsilon:
                    action = self.get_random_action()
                else:
                    action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                self.update_q_values(state, action, reward, next_state)
                total_reward += reward
                state = next_state

                if done:
                    total_rewards.append(total_reward)
                    mean_total_rewards = np.mean(total_rewards[-10:])

                    print(f"Episode: {episode + 1}/{num_episodes} \tTotal Reward: {total_reward} \tMean Total Rewards: {mean_total_rewards}")

                    if mean_total_rewards >= 495.0:
                        self.save_q_values()
                        return total_rewards
                    else:
                        break

        self.save_q_values()
        return total_rewards


    def play(self, num_episodes):
        self.load_q_values()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    print(f"Episode: {episode} \tTotal Reward: {total_reward}")


    def plot_rewards(self, total_rewards):
        plt.plot(range(len(total_rewards)), total_rewards, linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Total Rewards")
        plt.savefig(self.path_plot)


if __name__ == "__main__":

    # Hyperparameters
    GAMMA = 0.95
    ALPHA = 0.20
    EPSILON = 0.1

    PLAY = False
    EPISODES_TRAIN = 10000
    EPISODES_PLAY = 5
    
    env = gym.make("CartPole-v1")
    agent = Agent(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON)

    if not PLAY:
        total_rewards = agent.train(num_episodes=EPISODES_TRAIN)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)