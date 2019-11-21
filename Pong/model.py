import gym
import numpy as np
       
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense
from keras.optimizers import RMSprop

  
class DQN(Model):
    """
    Class representing the DQN model.
    """

    def __init__(self, img_shape, num_actions, lr):
        """
        Constructor of the DQN class.
            - img_shape : Input shape of the states
            - num_actions : Number of available actions
        """

        super(DQN, self).__init__()
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.lr = lr
 
        img = Input(shape=img_shape)
        x = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4))(img)
        x = Activation("relu")(x)
        x = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2))(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1))(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)
        out = Dense(self.num_actions)(x)
        
        self.model = Model(inputs=img, outputs=out)
        self.model.summary()
        self.model.compile(optimizer=RMSprop(lr=self.lr), loss="mse")


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