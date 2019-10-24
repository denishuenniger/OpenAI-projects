from keras.models import Model, Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam


class DQN(Model):

    def __init__(self, num_states, num_actions, lr):
        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr

        state = Input(shape=(num_states,))
        x = Dense(24, activation="relu")(state)
        action = Dense(num_actions, activation="softmax")(x)
        self.model = Model(inputs=state, outputs=action)
        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.lr))


    def fit(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)


    def predict(self, state):
        return self.model.predict(state)


    def update(self, other_model):
        self.model.set_weights(other_model.get_weights())


    def load(self, path):
        self.model.load_weights(path)


    def save(self, path):
        self.model.save_weights(path)