import torch as T
import torch.nn as nn
import torch.nn.functional as F
from protorl.networks.core import NetworkCore


class QHead(NetworkCore, nn.Module):
    def __init__(self, n_actions, input_dims=[256], hidden_layers=[512],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(hidden_layers) == 1, "Must supply 1 hidden layer size"
        self.fc1 = nn.Linear(*input_dims, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], n_actions)
        self.to(self.device)

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        q_values = self.fc2(f1)

        return q_values


class DuelingHead(NetworkCore, nn.Module):
    def __init__(self, n_actions, input_dims=[256], hidden_layers=[512],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, n_actions, input_dims=[256], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        mu = T.tanh(self.fc1(x))

        return mu


class MeanAndSigmaHead(NetworkCore, nn.Module):
    def __init__(self, n_actions, input_dims=[256], std_min=1e-6,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self, input_dims=[256], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = nn.Linear(*input_dims, 1)
        self.to(self.device)

    def forward(self, x):
        value = self.v(x)

        return value


class SoftmaxHead(NetworkCore, nn.Module):
    def __init__(self, n_actions, input_dims=[256], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        probs = F.softmax(self.probs(x), dim=1)

        return probs


class BetaHead(NetworkCore, nn.Module):
    def __init__(self, n_actions, input_dims=[256], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Linear(*input_dims, n_actions)
        self.beta = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, state):
        alpha = F.relu(self.alpha(state)) + 1.0
        beta = F.relu(self.beta(state)) + 1.0
        return alpha, beta


class DualValueHead(NetworkCore, nn.Module):
    def __init__(self, input_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_1 = nn.Linear(*input_dims, 1)
        self.value_2 = nn.Linear(*input_dims, 1)

    def forward(self, x):
        value_1 = self.value_1(x)
        value_2 = self.value_2(x)

        return value_1, value_2
