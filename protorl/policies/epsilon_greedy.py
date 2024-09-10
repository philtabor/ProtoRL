import numpy as np
import torch as T


class EpsilonGreedyPolicy:
    def __init__(self, n_actions=1, eps_start=1.0,
                 eps_dec=1e-5, eps_min=0.01,
                 config=None):
        if config:
            eps_start = config.eps_start
            eps_dec = config.eps_dec
            eps_min = config.eps_min
            n_actions = config.n_actions

        self.epsilon = eps_start
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(n_actions)]

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def __call__(self, q_values):
        if np.random.random() > self.epsilon:
            action = T.argmax(q_values, dim=-1).cpu().detach().numpy()
        else:
            action = np.random.choice(self.action_space)

        self.decrement_epsilon()
        return action
