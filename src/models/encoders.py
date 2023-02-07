import gym
import torch
import torch.nn.functional as F
from torch import nn


class EncoderModel(nn.Module):

    def __init__(self, conf):
        """
        Args:
            encoder_type: "cnn" or "fcn"
        """
        super().__init__()

        self.encoder = CnnEncoder(
            in_channels=conf.image_channels, cnn_depth=conf.cnn_depth)

        self.out_dim = self.encoder.out_dim

    def forward(self, x):
        y = self.encoder(x)
        return y


class CnnEncoder(nn.Module):
    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super(CnnEncoder, self).__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            activation(),
            nn.Flatten(),
        )

    def forward(self, x):
        B, T = x.shape[:2]
        x = torch.reshape(x, (-1, *x.shape[2:]))
        y = self.model(x)
        y = torch.reshape(y, (B, T, -1))
        return y


class FcnEncoder(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FcnEncoder, self).__init__()
        # will only work for discrete vars or vectors

        self.affine = nn.Linear(state_dim+action_dim, 128)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(128, 128) for _ in range(4)])
        self.head = nn.Linear(128, state_dim)

    def forward(self, states_actions):
        a = F.relu(self.affine(states_actions))
        for hidden_layer in self.hidden_layers:
            a = F.relu(hidden_layer(a))
        s_hat = self.head(a)
        return s_hat
