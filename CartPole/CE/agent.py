import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from keras.utils import to_categorical
from model import DNN


class Agent:
    """
    Class representing a learning agent acting in an environment.
    """

    def __init__(self, p, lr, game="CartPole-v1", mean_bound=5, reward_bound=495.0, save_model=10):
        """
        Constructor of the agent class.
            - game="CartPole-v1" : Name of the game environment
            - mean_bound=5 : Number of last acquired rewards considered for mean reward
            - reward_bound=495.0 : Reward acquired for completing an episode properly
            - save_model=10 : Interval for saving the model

            - p : Percentile for selecting training data
            - lr : Learning rate for the CE model
        """
        
        # Environment variables
        self.game = game
        self.env = gym.make(self.game)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Agent variables
        self.p = p * 100
        self.mean_bound = mean_bound
        self.reward_bound = reward_bound

        # DQN variables
        self.lr = lr
        self.model = DNN(self.num_states, self.num_actions, self.lr)
        self.save_model = save_model

        # File paths
        directory = os.path.dirname(__file__)
        self.path_model = os.path.join(directory, "models/dnn.h5")
        self.path_plot = os.path.join(directory, "plots/dnn.png")

        # Load model, if it already exists
        try:
            self.model.load(self.path_model)
        except:
            print(f"Model does not exist! Create new model...")

    
    def get_action(self, state):
        """
        Returns an action for a given state, based on the current policy.
            - state : Current state of the agent
        """

        state = state.reshape(1, -1)
        policy = self.model.predict(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        
        return action

    
    def sample(self, num_episodes):
        """
        Returns samples of state/action tuples for a given number of episodes.
            - num_episodes : Number of episodes to sample
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
                
                # Penalize agent if pole could not be balanced until end of episode.
                if done and reward < 499.0: reward = -100.0
                
                total_reward += reward

                if done:
                    total_reward += 100.0
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    
    def get_training_data(self, episodes, rewards):
        """
        Returns training data for the CE model.
            - episodes : List of state/action tuples
            - rewards : List of gained rewards
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
        y_train = to_categorical(y_train, num_classes=self.num_actions)

        return x_train, y_train, reward_bound


    def train(self, num_epochs, num_episodes, report_interval):
        """
        Trains the CE model for a given number of epochs and episodes. Outputting report information is controlled by a given time interval.
            - num_epochs : Number of epochs to train
            - num_episodes : Number of episodes to train
            - report_interval : Interval for outputting report information of training
        """

        total_rewards = []

        for epoch in range(1, num_epochs + 1):
            if epoch % self.save_model == 0:
                self.model.save(self.path_model)
            
            rewards, episodes = self.sample(num_episodes)
            x_train, y_train, reward_bound = self.get_training_data(episodes, rewards)

            mean_reward = np.mean(rewards)
            total_rewards.extend(rewards)
            mean_total_reward = np.mean(total_rewards[-self.mean_bound:])
            
            if epoch % report_interval == 0:
                print(
                    f"Epoch: {epoch + 1}/{num_epochs}"
                    f"\tMean Reward: {mean_reward : .2f}"
                    f"\tReward Bound: {reward_bound : .2f}")

                self.plot_rewards(total_rewards)

            if mean_total_reward > self.reward_bound:
                self.model.save(self.path_model)
            
            self.model.fit(x_train, y_train)

        self.model.save(self.path_model)

   
    def play(self, num_episodes):
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
                        f"Episode: {episode + 1}/{num_episodes}"
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
        plt.title("CE-Learning")
        plt.savefig(self.path_plot)


# Main program
if __name__ == "__main__":
    
    PLAY = True
    REPORT_INTERVAL = 10
    EPOCHS_TRAIN = 100
    EPISODES_TRAIN = 100
    EPISODES_PLAY = 5

    # Hyperparameters
    PERCENTILE = 0.8
    LEARNING_RATE = 0.001

    agent = Agent(p=PERCENTILE, lr=LEARNING_RATE)
    
    if not PLAY:
        agent.train(num_epochs=EPOCHS_TRAIN, num_episodes=EPISODES_TRAIN, report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)