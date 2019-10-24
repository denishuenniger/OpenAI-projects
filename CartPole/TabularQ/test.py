import gym

env = gym.make("CartPole-v1")

high = env.observation_space.high
low = env.observation_space.low

print(low, high)