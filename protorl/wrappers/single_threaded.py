import numpy as np
import gym


class SingleThreadedWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        try:
            obs, reward, done, trunc, info = self.env.step(action.item())
        # we may not have an extra dimension around the action
        except ValueError:
            obs, reward, done, trunc, info = self.env.step(action)

        return obs, reward, done, trunc, info


class BatchDimensionWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        observation = observation[np.newaxis, :]

        return observation
