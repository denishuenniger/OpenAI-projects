import cv2
import gym
import gym.spaces
import numpy as np

from collections import deque
 

class FireResetWrapper(gym.Wrapper):
    """
    Class representing a wrapper for environments where the user needs to press FIRE for starting the game.
    """
    
    def __init__(self, env=None):
        """
        Constructor for the fire-reset wrapper.

            - env=None : Current environment
        """
        
        super(FireResetWrapper, self).__init__(env)


    def step(self, action):
        """
        Takes an action in the current environment.

            - action : Action of the agent to take in the environment
        """

        return self.env.step(action)


    def reset(self):
        """
        Resets the current environment.
        """

        self.env.reset()
        state, _, done, _ = self.env.step(1)

        if done: self.env.reset()

        state, _, done, _ = self.env.step(2)
        
        if done: self.env.reset()
        
        return state


class MaxSkipWrapper(gym.Wrapper):
    """
    Class representing a wrapper for environments where input frames needs to be skippable and further selectable.
    """
    
    def __init__(self, env=None, skip_frame=4):
        """
        Constructor for the max-and-skip wrapper.

            - env=None : Current environment
            - skip_frame=4 : Number of how many frames should be skipped
        """
        
        super(MaxSkipWrapper, self).__init__(env)
        
        # Most recent raw observations (for Max-Pooling across time steps)
        self.frame_buffer = deque(maxlen=2)
        self.skip_frame = skip_frame

    
    def step(self, action):
        """
        Takes an action in the current environment.

            - action : Action of the agent to take in the environment
        """

        total_reward = 0.0
        
        for _ in range(self.skip_frame):
            state, reward, done, info = self.env.step(action)
            self.frame_buffer.append(state)
            total_reward += reward

            if done: break
        
        max_frame = np.max(np.stack(self.frame_buffer), axis=0)

        return max_frame, total_reward, done, info


    def reset(self):
        """
        Clears past frame buffer and initializes environment and buffer.
        """
        
        self.frame_buffer.clear()
        state = self.env.reset()
        self.frame_buffer.append(state)

        return state


class ProcessFrameWrapper(gym.ObservationWrapper):
    """
    Class representing a wrapper for environments where the input states need to be further processed.
    """
    
    def __init__(self, env=None):
        """
        Constructor for the process-frame wrapper.

            - env=None : Current environment
        """

        super(ProcessFrameWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)


    def observation(self, state):
        """
        Returns the processed state of the agent.

            - state : Current unprocessed state
        """

        return ProcessFrameWrapper.process(state)


    @staticmethod
    def process(frame):
        """
        Returns the processed frame.

            - frame : Current unprocessed frame
        """

        # Image size: 210 x 160 (RGB)
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.uint8)
        # Image size: 250 x 160 (RGB)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.uint8)
        else:
            assert False, "Unknown resolution!"

        # Adjust the image colors
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # Shrink image to 110 x 84 and cut to 84 x 84 
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        resized_img = resized_img[18:102, :]
        # Scale color values onto range 0...1
        scaled_img = np.reshape(resized_img, [1, 84, 84, 1]).astype(np.float32) / 255.0

        return scaled_img


def make_env(game):
    """
    Creates an environment for the agent and, based on the implemented wrappers, adds functionality.

        - game = Name of the environment to create
    """

    env = gym.make(game)
    env = MaxSkipWrapper(env)
    env = FireResetWrapper(env)
    env = ProcessFrameWrapper(env)

    return env
