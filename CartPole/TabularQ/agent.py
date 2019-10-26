import gym
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats import linregress

class Agent:

    def __init__(self, env, gamma, alpha,
                    epsilon, epsilon_min, epsilon_decay, on_policy):
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.q = defaultdict(lambda : [np.random.random() for _ in range(self.num_actions)])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.on_policy = on_policy

        # File paths
        dirname = os.path.dirname(__file__)
        self.path_model = os.path.join(dirname, "models/q.pickle")
        self.path_plot = os.path.join(dirname, "plots/q.png")


    def convert_state(self, state):
        return "_".join([str(np.round(x, 2)) for x in state])


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
            action = np.argmax(self.q[state])

        return action


    def get_q_value(self, state):
        if self.on_policy:
            q_value = self.q[state][self.get_action(state)]
        else:
            q_value = np.max(self.q[state])

        return q_value


    def update_q_values(self, state, action, reward, next_state, done):
        state = self.convert_state(state)
        next_state = self.convert_state(next_state)
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.get_q_value(next_state)
        
        q = self.q[state][action]
        q = (1 - self.alpha) * q + self.alpha * q_target

        self.q[state][action] = q


    def save_q_values(self):
        with open(self.path_model, "wb") as file:
            pickle.dump(dict(self.q), file)

    
    def load_q_values(self):
        try:
            with open(self.path_model, "rb") as file:
                self.q = defaultdict(lambda : [np.random.random() for _ in range(self.num_actions)], pickle.load(file))
        except:
            print("Model does not exist! Create new model...")


    def train(self, num_episodes, report_interval):
        total_rewards = []
        self.load_q_values()  
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(self.convert_state(state))
                next_state, reward, done, _ = self.env.step(action)
                
                if done and reward != 500.0: reward = -100.0
                
                self.update_q_values(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards)

                    if (episode + 1) % report_interval == 0:
                        print(f"Episode: {episode + 1}/{num_episodes}"
                            f"\tReward: {total_reward}"
                            f"\tMean Reward: {mean_reward}")

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
                    print(f"Episode: {episode + 1} \tReward: {total_reward}")
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
    GAMMA = 0.9
    ALPHA = 0.2
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    ON_POLICY = False

    PLAY = False
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 100000
    EPISODES_PLAY = 5
    
    env = gym.make("CartPole-v1") 
    agent = Agent(env,
                gamma=GAMMA,
                alpha=ALPHA,
                epsilon=EPSILON,
                epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY,
                on_policy=ON_POLICY)

    if not PLAY:
        total_rewards = agent.train(num_episodes=EPISODES_TRAIN,
                                    report_interval=REPORT_INTERVAL)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)