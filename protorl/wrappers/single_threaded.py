import numpy as np
import gymnasium as gym


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
    def __init__(self, env=None, scale_obs=True):
        super().__init__(env)
        shape = env.observation_space.shape
        # self.shape = (1, shape[2], shape[0], shape[1])
        self.shape = (1, *shape)
        self.scale_obs = scale_obs
        high = 1.0 if scale_obs else 255.0
        self.observation_space = gym.spaces.Box(low=0.0, high=high,
                                                shape=self.shape,
                                                dtype=np.float32)

    def observation(self, observation):
        return np.array([observation])
