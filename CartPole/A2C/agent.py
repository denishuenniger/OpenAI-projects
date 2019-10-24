import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from model import DNN


class Agent:

    def __init__(self, env, gamma, lr_actor, lr_critic):
        self.env = env
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.model = DNN(self.num_states, self.num_actions, self.lr_actor, self.lr_critic)

        # File paths
        directory = os.path.dirname(__file__)
        self.path_actor = os.path.join(directory, "models/actor.h5")
        self.path_critic = os.path.join(directory, "models/critic.h5")
        self.path_plot = os.path.join(directory, "plots/dnn.png")


    def get_action(self, state):
        policy = self.model.predict_actor(state)[0]
        action = np.random.choice(self.num_actions, p=policy)

        return action


    def update_policy(self, state, action, reward, next_state, done):
        values = np.zeros((1, 1))
        advantages = np.zeros((1, self.num_actions))
        
        value = self.model.predict_critic(state)[0]
        next_value = self.model.predict_critic(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        self.model.fit_actor(state, advantages)
        self.model.fit_critic(state, values)


    def train(self, num_episodes):
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

                self.update_policy(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    total_reward += 100.0
                    total_rewards.append(total_reward)
                    idx = -min(len(total_rewards), 10)
                    mean_total_rewards = np.mean(total_rewards[idx:])
                    
                    print(f"Episode: {episode + 1}/{num_episodes} \tTotal Reward: {total_reward} \tMean Total Rewards: {mean_total_rewards}")
                    
                    if mean_total_rewards >= 495.0:
                        self.model.save_actor(self.path_actor)
                        self.model.save_critic(self.path_critic)

                        return total_rewards
                    else:
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
                    print(f"Episode: {episode + 1} \t Reward: {reward}")
                    break


    def plot_rewards(self, total_rewards):
        plt.plot(range(len(total_rewards)), total_rewards, linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Total Rewards")
        plt.savefig(self.path_plot)


if __name__ == "__main__":

    # Hyperparameters
    GAMMA = 0.98
    LR_ACTOR = 1e-3
    LR_CRITIC = 5e-3

    PLAY = True
    EPISODES_TRAIN = 10000
    EPISODES_PLAY = 5

    env = gym.make("CartPole-v1")
    agent = Agent(env, gamma = GAMMA, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)
    
    if not PLAY:
        total_rewards = agent.train(num_episodes=EPISODES_TRAIN)
        agent.plot_rewards(total_rewards)
    else:
        agent.play(num_episodes=EPISODES_PLAY)