import torch as T
import torch.nn as nn
import torch.nn.functional as F
from protorl.networks.core import NetworkCore


class CriticBase(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, hidden_dims=[256, 256],
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.to(self.device)

    def forward(self, sa):
        state, action = sa
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class LinearBase(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, hidden_dims=[256, 256],
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.to(self.device)

    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        f2 = F.relu(self.fc2(f1))
        return f2


class AtariBase(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, channels=(32, 64, 64),
                 kernels=(8, 4, 3), strides=(4, 2, 1),
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)

        assert len(channels) == 3, "Must supply 3 channels for AtariBase"
        assert len(kernels) == 3, "Must supply 3 kernels for AtariBase"
        assert len(strides) == 3, "Must supply 3 strides for AtariBase"

        self.input_dims = input_dims
        self.channels = channels
        self.kernels = kernels
        self.strides = strides

        self.conv1 = nn.Conv2d(input_dims[0], channels[0],
                               kernels[0], strides[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1],
                               kernels[1], strides[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2],
                               kernels[2], strides[2])
        self.flat = nn.Flatten()

        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = self.flat(conv3)

        return conv_state


class LinearTanhBase(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, hidden_dims=[256, 256],
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.to(self.device)

    def forward(self, state):
        f1 = T.tanh(self.fc1(state))
        f2 = T.tanh(self.fc2(f1))
        return f2


class PPOCritic(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, hidden_dims=[128, 128],
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.v = nn.Linear(hidden_dims[1], 1)

        self.to(self.device)

    def forward(self, state):
        f1 = T.tanh(self.fc1(state))
        f2 = T.tanh(self.fc2(f1))
        v = self.v(f2)

        return v


class PPOActor(NetworkCore, nn.Module):
    def __init__(self, name, input_dims, n_actions, hidden_dims=[128, 128],
                 chkpt_dir='models'):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.alpha = nn.Linear(hidden_dims[1], n_actions)
        self.beta = nn.Linear(hidden_dims[1], n_actions)

        self.to(self.device)

    def forward(self, state):
        f1 = T.tanh(self.fc1(state))
        f2 = T.tanh(self.fc2(f1))
        alpha = F.relu(self.alpha(f2)) + 1.0
        beta = F.relu(self.beta(f2)) + 1.0

        return alpha, beta
