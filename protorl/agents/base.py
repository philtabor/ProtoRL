import torch as T
from protorl.utils.common import convert_arrays_to_tensors


class Agent:
    def __init__(self, memory, policy, gamma=0.99, tau=0.001):
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.tau = tau
        self.networks = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def store_transition(self, *args):
        self.memory.store_transition(*args)

    def sample_memory(self, mode='uniform'):
        memory_batch = self.memory.sample_buffer(mode=mode)
        memory_tensors = convert_arrays_to_tensors(memory_batch, self.device)
        return memory_tensors

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

    def update(self):
        raise(NotImplementedError)

    def choose_action(self, observation):
        raise(NotImplementedError)
