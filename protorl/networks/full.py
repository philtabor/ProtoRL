import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from protorl.networks.core import NetworkCore
from protorl.utils.common import calculate_conv_output_dims


class PPOAtariNetwork(NetworkCore, nn.Module):
    def __init__(self, input_dims, n_actions,
                 channels=(32, 64, 64),
                 kernels=(8, 4, 3),
                 strides=(4, 2, 1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims[0], channels[0], kernels[0], strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernels[1], strides[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernels[2], strides[2]),
            nn.ReLU(),
            nn.Flatten(),
        )

        input_dims = calculate_conv_output_dims()

        self.critic = nn.Sequential(nn.Linear(*input_dims, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)
        )
        self.actor = nn.Sequential(nn.Linear(*input_dims, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, n_actions),
                                   nn.Softmax(dim=1),
        )

        self.to(self.device)

        self.init_weights()

    def init_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in self.actor.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in self.critic.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.cnn(x)
        pi = self.actor(x)
        v = self.critic(x)
        return pi, v
