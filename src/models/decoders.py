import torch
import torch.nn as nn


class DecoderModel(nn.Module):
    def __init__(self, features_dim, conf):
        super().__init__()
        self.decoder = ConvDecoder(
            features_dim, out_channels=conf.image_channels, cnn_depth=conf.cnn_depth)
        self.image_weight = conf.image_weight

    def forward(self, latents, pixels):
        loss_image, decoded_img = self.decoder(
            latents, pixels)
        loss = self.image_weight * loss_image

        return loss, loss_image, decoded_img


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 out_channels=3,
                 cnn_depth=32,
                 mlp_layers=0,
                 layer_norm=True,
                 activation=nn.ELU
                 ):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [
                nn.Linear(in_dim, d * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm
            layers = [
                nn.Linear(in_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]

        self.model = nn.Sequential(
            nn.Flatten(0, 1),
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),  # type: ignore
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channels, kernels[3], stride),

        )

    def forward(self, features, target):
        decoded_img = self.model(features)
        decoded_img = torch.reshape(
            decoded_img, target.shape)
        loss = 0.5 * torch.square(decoded_img -
                                  target).sum(dim=[-1, -2, -3])  # MSE

        return loss, decoded_img
