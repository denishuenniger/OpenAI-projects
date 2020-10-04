from agent import *
from model import *

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
        agent.train(num_epochs=EPOCHS_TRAIN,
                    num_episodes=EPISODES_TRAIN, report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)
