"""Simple implementation of VAE.

very simple
"""

from __future__ import annotations
from dataclasses import dataclass

import os

import torch
from torch import nn

from pml_vqvae.visuals import show

from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.nnutils import ResidualBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaselineVAEConfig:
    kld_weight: float  # for starter experiments we've used 0.00025
    hidden_dimension: int
    latent_dimension: int

    name: str = "VAE"


class BaselineVAE(PML_model):
    def __init__(self, config: BaselineVAEConfig):
        super().__init__()

        self.config = config

        hidden_dim = config.hidden_dimension
        latent_dim = config.latent_dimension

        self.encoder = nn.Sequential(
            # Downsampling
            nn.Conv2d(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            # Residuals
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            # Compression
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=latent_dim * 2,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            # Decompress
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            # Residuals
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            # Upsampling
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
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

    def loss_fn(self, model_outputs, target):
        """Computes the VAE loss which consist of reconstruction loss and KL
        which consist of reconstruction loss and KL
        divergence between prior distribution z ~ N(0, I) and posterior
        distribution z|x ~ N(mean, exp(logvar)).

        Also stolen from the internet somewhere
        """
        reconstruction, mean, logvar = model_outputs

        reconstruction_loss = torch.mean((target - reconstruction) ** 2)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=(1, 2, 3)),
            dim=0,
        )

        loss = reconstruction_loss + self.config.kld_weight * kld_loss

        self.batch_stats = {
            "Loss": loss.detach().cpu().item(),
            "Reconstruction loss": reconstruction_loss.item(),
            "KL divergence loss": kld_loss.item(),
            "KL divergence loss (weighted)": self.config.kld_weight * kld_loss.item(),
        }

        return loss

    def backward(self, loss):
        loss.backward()

    def visualize_output(self, output):
        return output[0]

    def name(self):
        return "BaselineVariationalAutoencoder"
