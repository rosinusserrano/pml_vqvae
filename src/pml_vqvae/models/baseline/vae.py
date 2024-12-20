"""Simple implementation of VAE.

very simple
"""

from __future__ import annotations

import os

import torch
from torch import nn

from pml_vqvae.visuals import show

from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.nnutils import ResidualBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineVariationalAutoencoder(PML_model):
    def __init__(self):
        super().__init__()

        hidden_dim = 128
        latent_dim = 2

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

        loss = reconstruction_loss + 0.00025 * kld_loss

        self.batch_stats = {
            "Loss": loss[0].detach().cpu().item(),
            "Reconstruction loss": loss[1].item(),
            "KL divergence loss": loss[2].item(),
        }

        return loss

    def backward(self, loss):
        loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output[0],
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "BaselineVariationalAutoencoder"
