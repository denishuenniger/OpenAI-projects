import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from model import DNN


class Agent:
    """
    Represents an agent to train in the given environment.
        env           = Environment of the agent
        num_states    = Number of dimensions of the state space
        num_actions   = Number of dimensions of the action space
        lr            = Learning rate for the Neural Network model
        p             = Percentile of generating training data
        model         = Neural Network model
    """

    def __init__(self, env, p, lr):
        """
        Constructor of the Agent class.
        """

        # Hyperparamters
        self.p = p * 100
        self.lr = lr

        # Agent variables
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.model = DNN(self.num_states, self.num_actions, self.lr)

        # File paths
        directory = os.path.dirname(__file__)
        self.path_model = os.path.join(directory, "models/dnn.h5")
        self.path_plot = os.path.join(directory, "plots/dnn.png")

    
    def predict_action(self, state):
        """
        Predicts an action from the current policy.
            state = Current state of the agent
        """

        state = state.reshape(1, -1)
        policy = self.model.predict(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        
        return action

    
    def sample(self, num_episodes):
        """
        Samples an running agent for a given number of episodes.
            num_episodes = Number of episodes
        """

        episodes = [[] for _ in range(num_episodes)]
        rewards = [0.0 for _ in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.predict_action(state)
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
        y_train = to_categorical(y_train, num_classes=self.num_actions)

        return x_train, y_train, reward_bound


    def train(self, num_epochs, num_episodes):
        """
        Trains the Neural Network model.
            num_epochs      = Number of training epochs
            num_episodes    = Number of episodes to sample 
        """

        try:
            self.model.load(self.path_model)
        except:
            print(f"File does not exist! Create new file...")

        total_rewards = []

        for epoch in range(num_epochs):
            rewards, episodes = self.sample(num_episodes)
            x_train, y_train, reward_bound = self.get_training_data(episodes, rewards)
            mean_reward = np.mean(rewards)
            total_rewards.extend(rewards)
            
            self.model.fit(x_train, y_train)
            print(f"Epoch: {epoch + 1}/{num_epochs} \tMean Reward: {mean_reward} \tReward Bound: {reward_bound}")

        self.model.save(self.path_model)
        
        return total_rewards

   
    def test(self, num_episodes, render):
        """
        Tests the trained agent for a given number of episodes.
            num_episodes    = Number of episodes to test
            render          = Flag whether training is displayed or not
        """

        self.model.load(self.path_model)

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()

                action = self.predict_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    print(f"Episode: {episode + 1}/{num_episodes} \tReward: {total_reward}")
                    break


    def plot_rewards(self, total_rewards):
        plt.plot(range(len(total_rewards)), total_rewards, linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Total Rewards")
        plt.savefig(self.path_plot)


if __name__ == "__main__":
    
    # Hyperparameters
    PERCENTILE = 0.75
    LEARNING_RATE = 1e-3

    env = gym.make("CartPole-v1")
    agent = Agent(env, p=PERCENTILE, lr=LEARNING_RATE)
    total_rewards = agent.train(num_epochs=10, num_episodes=100)
    agent.plot_rewards(total_rewards)
    
    input("PLAY?")
    agent.test(num_episodes=10, render=True)