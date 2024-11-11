"""Simple implementation of VAE.

very simple
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show_image_grid

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pml_vqvae.baseline.pml_model_interface import PML_model


def conv_block(inc: int, outc: int) -> nn.Module:
    """Create a simple 3-layered conv block."""
    return nn.Sequential(
        nn.Conv2d(inc, outc, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(outc, outc, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(outc, outc, 3, padding=1),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    """Create residual layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a residual block."""
        super().__init__()
        self.conv_block = conv_block(in_channels, out_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection."""
        out = self.conv_block(x)
        skip = self.skip_conv(x)
        return out + skip


class BaselineVariationalAutoencoder(PML_model):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 128x128x3 -> 64x64x8
            ResidualBlock(3, 8),
            nn.MaxPool2d(2, 2),
            # 64x64x8 -> 32x32x16
            ResidualBlock(8, 16),
            nn.MaxPool2d(2, 2),
            # 32x32x16 -> 28x28x64
            nn.Conv2d(16, 64, 5),
            nn.ReLU(),
            # 28x28x64 -> 28x28x2
            nn.Conv2d(64, 16, 1),
        )
        self.decoder = nn.Sequential(
            # 28x28x1 -> 32x32x64
            nn.ConvTranspose2d(8, 64, 5),
            nn.ReLU(),
            # 32x32x64 -> 32x32x16
            ResidualBlock(64, 16),
            # 32x32x16 -> 64x64x16
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.ReLU(),
            # 64x64x16 -> 64x64x8
            ResidualBlock(16, 8),
            # 64x64x8 -> 128x128x8
            nn.ConvTranspose2d(8, 8, 4, 2, 1),
            nn.ReLU(),
            # 128x128x8 -> 128x128x3
            ResidualBlock(8, 3),
        )

    def reparameterization(self, encoder_output: torch.Tensor):
        """Splits the output of the encoder into logvar and mean and then samples
        using the reparameterization trick"""
        batch_size, n_feats, height, width = encoder_output.shape

        assert (
            n_feats % 2 == 0
        ), """Use even number of output features otherwise
        one can't split them into mean and variance"""

        z = torch.randn((batch_size, n_feats // 2, height, width)).to(DEVICE)
        mean = encoder_output[:, : (n_feats // 2), ...]
        logvar = encoder_output[:, (n_feats // 2) :, ...]
        z_hat = mean + torch.exp(0.5 * logvar) * z

        return z_hat, mean, logvar

    def forward(self, x):
        z = self.encoder(x)
        z_hat, mean, logvar = self.reparameterization(z)
        x_hat = self.decoder(z_hat)

        return x_hat, mean, logvar

    @staticmethod
    def loss_fn():
        return loss_function

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def name(self):
        return "BaselineVariationalAutoencoder"


def loss_function(reconstruction, mean, logvar, original):
    """Computes the VAE loss which consist of reconstruction loss and KL
    which consist of reconstruction loss and KL
    divergence between prior distribution z ~ N(0, I) and posterior
    distribution z|x ~ N(mean, exp(logvar)).
    """
    reconstruction_loss = torch.mean((original - reconstruction) ** 2)

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=(1, 2, 3)),
        dim=0,
    )

    loss = reconstruction_loss + 0.001 * kld_loss

    return loss, reconstruction_loss.detach(), -kld_loss.detach()
