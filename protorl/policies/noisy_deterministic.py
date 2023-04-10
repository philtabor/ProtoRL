import numpy as np
import torch as T


class NoisyDeterministicPolicy:
    def __init__(self, n_actions, noise=0.1, min_action=-1, max_action=1):
        self.noise = noise
        self.min_action = min_action
        self.max_action = max_action

    def __call__(self, mu, scale=None, noise_bounds=None):
        scale = scale or self.noise
        mu = mu.detach()
        # Some environments have max action outside the range of +/- 1 that we
        # get from the tanh activation function
        mu *= T.abs(T.tensor(self.max_action))
        noise = T.tensor(np.random.normal(scale=scale), dtype=T.float)
        if noise_bounds:
            noise = T.clamp(noise, noise_bounds[0], noise_bounds[1])
        mu = mu + noise
        mu = T.clamp(mu, self.min_action, self.max_action)
        return mu
