from keras.models import Model, Sequential, Input
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class DQN(Model):
    """
    Class representing the DQN model.
    """

    def __init__(self, num_states, num_actions, lr):
        """
        Constructor of the DQN class.
            - img_shape : Input shape of the states
            - num_actions : Number of available actions
        """

        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr

        state = Input(shape=(num_states,))
        x = Dense(32)(state)
        x = Activation("relu")(x)
        action = Dense(num_actions)(x)
        self.model = Model(inputs=state, outputs=action)
        self.model.summary()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))


    def fit(self, states, q_values):
        """
        Fits the DQN model based on the given states and correspoonding Q-values.
            - states : States of the agent
            - q_values : Q-values for available actions and given states
        """

        self.model.fit(states, q_values, verbose=0)


    def predict(self, state):
        """
        Predicts the q-values from the given states.
            - state : Current state of the agent
        """

        return self.model.predict(state)


    def update(self, other_model):
        """
        Synchronizes the target model with the DQN model.
            - other_model : Current DQN model
        """

        self.model.set_weights(other_model.get_weights())


    def load(self, path):
        """
        Loads a DQN model from the given path.
            - path : File path to the DQN model
        """

        self.model.load_weights(path)


    def save(self, path):
        """
        Saves the DQN model to a given path.
            - path : File path to the save location
        """
        
        self.model.save_weights(path)