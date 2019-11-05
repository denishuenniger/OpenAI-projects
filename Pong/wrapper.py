import cv2
import gym
import gym.spaces
import numpy as np

from collections import deque
 

class FireResetEnv(gym.Wrapper):
    
    def __init__(self, env=None):
        """For environments where the user needs to press FIRE for starting the game."""
        
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3


    def step(self, action):
        return self.env.step(action)


    def reset(self):
        self.env.reset()
        state, _, done, _ = self.env.step(1)

        if done: self.env.reset()

        state, _, done, _ = self.env.step(2)
        
        if done: self.env.reset()
        
        return state


class MaxAndSkipEnv(gym.Wrapper):
    
    def __init__(self, env=None, skip_frame=4):
        """Returns only every 4-th frame."""
        
        super(MaxAndSkipEnv, self).__init__(env)
        
        # Most recent raw observations (for Max-Pooling across time steps)
        self.frame_buffer = deque(maxlen=2)
        self.skip_frame = skip_frame

    
    def step(self, action):
        total_reward = 0.0
        done = None
        
        for _ in range(self.skip_frame):
            state, reward, done, info = self.env.step(action)
            self.frame_buffer.append(state)
            total_reward += reward

            if done: break
        
        max_frame = np.max(np.stack(self.frame_buffer), axis=0)
        return max_frame, total_reward, done, info


    def reset(self):
        """Clears past frame buffer and initializes environment and buffer."""
        
        self.frame_buffer.clear()
        state = self.env.reset()
        self.frame_buffer.append(state)

        return state


class ProcessFrame(gym.ObservationWrapper):
    
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)


    def observation(self, state):
        return ProcessFrame.process(state)


    @staticmethod
    def process(frame):
        # Image size: 210 x 160 (RGB)
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.uint8)
        # Image size: 250 x 160 (RGB)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.uint8)
        else:
            assert False, "Unknown resolution."

        # Adjust the image colors
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # Shrink image to 110 x 84 and cut to 84 x 84 
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        resized_img = resized_img[18:102, :]
        # Scale color values onto range 0...1
        scaled_img = np.reshape(resized_img, [1, 84, 84, 1]).astype(np.float32) / 255.0

        return scaled_img

def make_env(game):
    env = gym.make(game)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame(env)

    return env
