import torch as T
import torch.nn as nn
import torch.nn.functional as F
from protorl.networks.core import NetworkCore


class QHead(NetworkCore, nn.Module):
    def __init__(self, name,  n_actions,
                 input_dims=[256], hidden_layers=[512], chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)

        assert len(hidden_layers) == 1, "Must supply 1 hidden layer size"
        self.fc1 = nn.Linear(*input_dims, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], n_actions)
        self.to(self.device)

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        q_values = self.fc2(f1)

        return q_values


class DuelingHead(NetworkCore, nn.Module):
    def __init__(self, name, n_actions,
                 input_dims=[256], hidden_layers=[512], chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)

        assert len(hidden_layers) == 1, "Must supply 1 hidden layer size"
        self.fc1 = nn.Linear(*input_dims, hidden_layers[0])
        self.V = nn.Linear(hidden_layers[0], 1)
        self.A = nn.Linear(hidden_layers[0], n_actions)

        self.to(self.device)

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        V = self.V(f1)
        A = self.A(f1)

        return V, A


class DeterministicHead(NetworkCore, nn.Module):
    def __init__(self, name, n_actions, chkpt_dir='models', input_dims=[256]):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        mu = T.tanh(self.fc1(x))

        return mu


class MeanAndSigmaHead(NetworkCore, nn.Module):
    def __init__(self, name, n_actions,
                 input_dims=[256], chkpt_dir='models', std_min=1e-6):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.std_min = std_min
        self.mu = nn.Linear(*input_dims, n_actions)
        self.sigma = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = T.clamp(sigma, min=self.std_min, max=1)

        return mu, sigma


class ValueHead(NetworkCore, nn.Module):
    def __init__(self, name, input_dims=[256], chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.v = nn.Linear(*input_dims, 1)
        self.to(self.device)

    def forward(self, x):
        value = self.v(x)

        return value


class SoftmaxHead(NetworkCore, nn.Module):
    def __init__(self, name, n_actions,
                 input_dims=[256], chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.probs = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        probs = F.softmax(self.probs(x), dim=1)

        return probs


class BetaHead(NetworkCore, nn.Module):
    def __init__(self, name, n_actions,
                 input_dims=[256], chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.alpha = nn.Linear(*input_dims, n_actions)
        self.beta = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, state):
        alpha = F.relu(self.alpha(state)) + 1.0
        beta = F.relu(self.beta(state)) + 1.0
        return alpha, beta
