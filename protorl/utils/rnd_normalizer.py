import torch as T
import numpy as np


class ObsNormalizer:
    def __init__(self, shape, epsilon=1e-8):
        self.running_mean = T.zeros(shape, dtype=T.float, device='cuda:0')
        self.running_variance = T.ones(shape, dtype=T.float, device='cuda:0')

        self.count = 0
        self.epsilon = epsilon

    def update(self, observation):
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        delta2 = observation - self.running_mean
        self.running_variance += delta * delta2

    def normalize(self, observation):
        return (observation - self.running_mean) / (T.sqrt(self.running_variance/self.count) + self.epsilon)


class RNormalizer:
    def __init__(self, shape, epsilon=1e-4):
        self.var = np.ones(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)
        self.epsilon = epsilon

    def update(self, observation):
        self.var = self.var * 0.99 + observation.var() * 0.01
        self.std = np.sqrt(self.var)

    def normalize(self, x):
        return np.clip(x / (self.std + self.epsilon), 0, 1)
