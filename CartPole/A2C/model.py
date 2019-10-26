from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop


class DNN(Model):

    def __init__(self, num_states, num_actions, lr_actor, lr_critic):
        super(DNN, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        state = Input(shape=(num_states,))
        x = Dense(64, activation="relu")(state)

        actor_x = Dense(32, activation="relu")(x)
        actor_out = Dense(self.num_actions)(actor_x)
        self.actor = Model(inputs=state, outputs=actor_out)
        self.actor.summary()
        self.actor.compile(loss="mse", optimizer=Adam(lr=self.lr_actor))

        critic_x = Dense(32, activation="relu")(x)
        critic_out = Dense(1)(critic_x)
        self.critic = Model(inputs=state, outputs=critic_out)
        self.critic.summary()
        self.critic.compile(loss="mse", optimizer=Adam(lr=self.lr_critic))


    def fit_actor(self, state, action):
        self.actor.fit(state, action, verbose=0)


    def fit_critic(self, state, value):
        self.critic.fit(state, value, verbose=0)


    def predict_actor(self, state):
        return self.actor.predict(state)


    def predict_critic(self, state):
        return self.critic.predict(state)


    def load_actor(self, path):
        self.actor.load_weights(path)

    
    def load_critic(self, path):
        self.critic.load_weights(path)


    def save_actor(self, path):
        self.actor.save_weights(path)


    def save_critic(self, path):
        self.critic.save_weights(path)