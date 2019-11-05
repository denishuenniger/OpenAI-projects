import gym
import numpy as np
       
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense
from keras.optimizers import RMSprop

  
class DQN(Model):

    def __init__(self, img_shape, num_actions, lr):
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
        self.model.fit(states, q_values, verbose=0)


    def predict(self, state):
        return self.model.predict(state)


    def update(self, other_model):
        self.model.set_weights(other_model.get_weights())


    def load(self, path):
        self.model.load_weights(path)


    def save(self, path):
        self.model.save_weights(path)   