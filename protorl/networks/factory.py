from protorl.core import NetworkCore
import torch.nn as nn


class LinearNetworkFactory(NetworkCore, nn.Module):
    def __init__(self, input_dims, hidden_layer_dims, activations,
                 name='network', chkpt_dir='models/'):
        super(LinearNetworkFactory, self).__init__()
        c_dim = input_dims
        self.activations = activations
        self.layers = nn.ModuleList()
        for dim in hidden_layer_dims:
            self.layers.append(nn.Linear(*c_dim, dim))
            c_dim = [dim]

    def forward(self, x):
        for idx, layer in enumerate(self.layers[:-1]):
            x = self.activations[idx](layer(x))
        out = self.activations[-1](self.layers[-1](x))

        return out


class ConvNetworkFactor(NetworkCore, nn.Module):
    def __init__(self, input_channels, channels, kernels, strides, activations):
        c_dim = input_channels
        self.activations = activations
        self.layers = nn.ModuleList()
        for ch, k, s in zip(channels, kernels, strides):
            self.layers.append(nn.Conv2d(in_channels=c_dim, out_channels=ch,
                                         kernel_size=k, stride=s))
            c_dim = ch
        self.layers.append(nn.Flatten())

    def forward(self, x):
        for idx, layer in enumerate(self.layers[:-1]):
            x = self.activations[idx](layer(x))
        out = self.activations[-1](self.layers[-1](x))

        return out
