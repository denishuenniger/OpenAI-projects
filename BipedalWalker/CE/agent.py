import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from model import DNN


class Agent:
    """
    Represents an agent to train in the given environment.
        env           = Environment of the agent
        num_states    = Number of dimensions of the state space
        num_actions   = Number of dimensions of the action space
        p             = Percentile of generating training data
        learning_rate = Learning rate for the Neural Network model
        model         = Neural Network model
    """

    def __init__(
        self, env,
        p, epsilon, epsilon_min, epsilon_decay,
        learning_rate):
        """
        Constructor of the Agent class.
        """

        # Hyperparamters
        self.p = p * 100
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Agent variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.model = DNN(self.num_states, self.num_actions, self.learning_rate)

        # File paths
        directory = os.path.dirname(__file__)
        self.path_model = os.path.join(directory, "models/dnn.h5")
        self.path_plot = os.path.join(directory, "plots/dnn.png")

    
    def reduce_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay

        if epsilon >= self.epsilon_min:
            self.epsilon = epsilon
        else:
            self.epsilon = self.epsilon_min


    def get_action(self, state):
        """
        Predicts an action from the current policy.
            state = Current state of the agent
        """

        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = state.reshape(1, self.num_states)
            action = self.model.predict(state)[0]

        return action

    
    def sample(self, num_episodes):
        """
        Samples a running agent for a given number of episodes.
            num_episodes = Number of episodes
        """

        episodes = [[] for _ in range(num_episodes)]
        rewards = [0.0 for _ in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episodes[episode].append((state, action))
                state = next_state
                
                total_reward += reward

                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    
    def get_training_data(self, episodes, rewards):
        """
        Creates data to train the Neural Network model, based on the reward for a given episode.
            episodes    = List of episodes to sample from
            rewards     = List of rewards for sampled episodes
        """

        x_train, y_train = [], []
        reward_bound = np.percentile(rewards, self.p)

        for episode, reward in zip(episodes, rewards):
            if reward >= reward_bound:
                states = [step[0] for step in episode]
                actions = [step[1] for step in episode]
                x_train.extend(states)
                y_train.extend(actions)
        
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        return x_train, y_train, reward_bound


    def train(self, num_epochs, num_episodes, report_interval):
        """
        Trains the Neural Network model.
            num_epochs      = Number of training epochs
            num_episodes    = Number of episodes to sample
            report_interval = Episode interval for reporting
        """

        try:
            self.model.load(self.path_model)
        except:
            print(f"Model does not exist! Create new model...")

        total_rewards = []

        for epoch in range(num_epochs):
            rewards, episodes = self.sample(num_episodes)
            self.reduce_epsilon()
            x_train, y_train, reward_bound = self.get_training_data(episodes, rewards)
            mean_reward = np.mean(rewards)
            total_rewards.extend(rewards)
            
            if (epoch + 1) % report_interval == 0:
                print(
                    f"Epoch: {epoch + 1}/{num_epochs}"
                    f"\tMean Reward: {mean_reward : .2f}"
                    f"\tReward Bound: {reward_bound : .2f}")
            
            self.model.fit(x_train, y_train)

        self.model.save(self.path_model)
        return total_rewards

   
    def play(self, num_episodes):
        """
        Tests the trained agent for a given number of episodes.
            num_episodes    = Number of episodes to test
        """

        self.model.load(self.path_model)

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
                        f"Episode: {episode + 1}/{num_episodes}"
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
        plt.title("CE-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":
    
    # Hyperparameters
    PERCENTILE = 0.8
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    LEARNING_RATE = 0.001

    PLAY = False
    REPORT_INTERVAL = 10
    EPOCHS_TRAIN = 100
    EPISODES_TRAIN = 100
    EPISODES_PLAY = 5

    env = gym.make("BipedalWalker-v2")

    agent = Agent(
        env=env,
        p=PERCENTILE,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        learning_rate=LEARNING_RATE)
    
    if not PLAY:
        total_rewards = agent.train(
            num_epochs=EPOCHS_TRAIN,
            num_episodes=EPISODES_TRAIN,
            report_interval=REPORT_INTERVAL)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)