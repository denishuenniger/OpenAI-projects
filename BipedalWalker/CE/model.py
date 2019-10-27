from keras.models import Model, Sequential, Input
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class DNN(Model):

    def __init__(self, num_states, num_actions, learning_rate):
        super(DNN, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        state = Input(shape=(self.num_states,))
        x = Dense(512)(state)
        x = Activation("relu")(x)
        action = Dense(self.num_actions)(x)

        self.model = Model(inputs=state, outputs=action)
        self.model.summary()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))


    def fit(self, state, action):
        self.model.fit(state, action, verbose=0)


    def predict(self, state):
        return self.model.predict(state)


    def load(self, path):
        self.model.load_weights(path)

    
    def save(self, path):
        self.model.save_weights(path)