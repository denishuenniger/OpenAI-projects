from agent import *
from model import *

if __name__ == "__main__":

    PLAY = True
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5

    # Hyperparameters
    GAMMA = 0.95
    ALPHA = 0.2
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98

    agent = Agent(
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY)

    if not PLAY:
        agent.train(num_episodes=EPISODES_TRAIN,
                    report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)
