import numpy as np
import torch.nn as nn
from protorl.networks.core import NetworkCore
from protorl.utils.common import calculate_conv_output_dims


class RandomCNN(NetworkCore, nn.Module):
    def __init__(self, input_dims, input_channels=1, output_size=512):
        super().__init__()
        output_dims = calculate_conv_output_dims(input_dims)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(*output_dims, output_size)
        )
        self.to(self.device)
        self.network.apply(self.init_weights)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.network(x)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

class PredictorCNN(NetworkCore, nn.Module):
    def __init__(self, input_dims, input_channels=1, output_size=512):
        super().__init__()
        output_dims = calculate_conv_output_dims(input_dims)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(*output_dims, output_size)
        )
        self.to(self.device)
        self.network.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

class PPONetwork(NetworkCore, nn.Module):
    def __init__(self, input_dims, n_actions,
                 channels=(32, 64, 64),
                 kernels=(8, 4, 3),
                 strides=(4, 2, 1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv_dims = calculate_conv_output_dims(input_dims, channels, kernels, strides)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims[0], channels[0], kernels[0], strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernels[1], strides[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernels[2], strides[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(*conv_dims, 512),
            nn.ReLU(),
        )
        self.ext_critic = nn.Sequential(nn.Linear(512, 1))
        self.int_critic = nn.Sequential(nn.Linear(512, 1))

        self.actor = nn.Sequential(nn.Linear(512, n_actions),
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

        for m in self.ext_critic.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in self.int_critic.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.cnn(x)
        pi = self.actor(x)
        ext_v = self.ext_critic(x)
        int_v = self.int_critic(x)
        return pi, ext_v, int_v
