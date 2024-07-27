import torch as T


class Learner:
    def __init__(self, gamma=0.99, tau=0.001):
        self.gamma = gamma
        self.tau = tau
        self.networks = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def save_models(self):
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError

    def update_network_parameters(self, src, dest, tau=None):
        if tau is None:
            tau = self.tau
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def update(self, *args):
        raise NotImplementedError
