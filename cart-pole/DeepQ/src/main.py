from agent import *

if __name__ == "__main__":

    # Choose whether to play or to train
    PLAY = True
    REPORT_INTERVAL = 100
    EPISODES_TRAIN = 1000
    EPISODES_PLAY = 5

    # Hyperparameters
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 128
    ALPHA = 0.2
    GAMMA = 0.95
    EPSILON = 0.1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.98
    LEARNING_RATE = 0.001

    agent = Agent(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        lr=LEARNING_RATE)

    if not PLAY:
        agent.train(num_episodes=EPISODES_TRAIN,
                    report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)
