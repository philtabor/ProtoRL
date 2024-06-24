import torch as T


class Actor:
    def __init__(self, policy, tau=0.001, device=None):
        self.policy = policy
        self.tau = tau

        self.networks = []
        self.device = device or T.device('cuda:0' if T.cuda.is_available()
                                         else 'cpu')

    def save_models(self):
        for network in self.networks:
            for x in network:
                x.save_checkpoint()

    def load_models(self):
        for network in self.networks:
            for x in network:
                x.load_checkpoint()

    def update_network_parameters(self, src, dest, tau=None):
        if tau is None:
            tau = self.tau
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def choose_action(self, observation):
        raise NotImplementedError
