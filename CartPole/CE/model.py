from keras.models import Model, Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam


class DNN(Model):

    def __init__(self, num_states, num_actions, lr):
        super(DNN, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr

        state = Input(shape=(self.num_states,))
        x = Dense(64, activation="relu")(state)
        action = Dense(self.num_actions, activation="softmax")(x)

        self.model = Model(inputs=state, outputs=action)
        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.lr))


    def fit(self, state, action):
        self.model.fit(state, action, verbose=0)


    def predict(self, state):
        return self.model.predict(state)


    def load(self, path):
        self.model.load_weights(path)

    
    def save(self, path):
        self.model.save_weights(path)