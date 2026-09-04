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
        self.critic = nn.Sequential(nn.Linear(512, 1)
        )
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


class R2D2Network(NetworkCore, nn.Module):
    def __init__(self, input_dims, n_actions,
                 input_channels=4,
                 channels=(32, 64, 64),
                 kernels=(8, 4, 3),
                 strides=(4, 2, 1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_actions = n_actions
        conv_dims = calculate_conv_output_dims(input_dims, channels, kernels, strides)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernels[0], strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernels[1], strides[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernels[2], strides[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(*conv_dims, 512),
            nn.ReLU(),
        )
        recurrent_input_size = 512 + int(n_actions) + 1
        self.recurrent = nn.LSTM(input_size=recurrent_input_size, hidden_size=128, batch_first=True)
        self.V = nn.Sequential(nn.Linear(128, 1))
        self.A = nn.Sequential(nn.Linear(128, n_actions))

        self.to(self.device)

        self.init_weights()

    def init_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in self.A.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in self.V.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, hx, cx, prev_a, prev_r):
        x = self.cnn(x)
        prev_a_one_hot = F.one_hot(prev_a.type(T.long), num_classes=self.n_actions)

        # single action
        if prev_a_one_hot.shape == T.zeros(self.n_actions).shape:
            prev_a_one_hot = prev_a_one_hot.unsqueeze(0)
            prev_r = prev_r.reshape((1,1))

        if prev_a.shape > T.zeros(1).shape:
            shape = (*prev_a.shape, 512)
            x = x.reshape(*shape)
        # recurrent_input = T.cat([x, prev_a_one_hot, prev_r.reshape((1,1))], dim=-1)
        # print(f"shapes -> x: {x.shape} prev_a_one_hot: {prev_a_one_hot.shape} prev_r: {prev_r.shape}")
        recurrent_input = T.cat([x, prev_a_one_hot, prev_r], dim=-1)
        if prev_a.shape > T.zeros(1).shape:
            recurrent_input = recurrent_input.unsqueeze(1)

        recurrent_out, (hx_new, cx_new) = self.recurrent(recurrent_input, (hx, cx))

        V = self.V(recurrent_out)
        A = self.A(recurrent_out)
        return V, A, (hx_new, cx_new)
