from agent import *

if __name__ == "__main__":

    # Choose whether to play or to train
    PLAY = False
    EPISODES_TRAIN = 10000
    EPISODES_PLAY = 15
    REPORT_INTERVAL = 10

    # Hyperparameters
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99
    BUFFER_SIZE = 50000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00025

    agent = Agent(
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, lr=LEARNING_RATE)

    if not PLAY:
        agent.train(num_episodes=EPISODES_TRAIN,
                    report_interval=REPORT_INTERVAL)
    else:
        agent.play(num_episodes=EPISODES_PLAY)
