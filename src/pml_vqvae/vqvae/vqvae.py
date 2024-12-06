import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show
from pml_vqvae.nnutils import ResidualBlock


class VQVAE(PML_model):
    def __init__(self):
        hidden_chan = 128
        latent_chan = 256  # D in the paper
        num_codes = 512  # K in the paper
        self.beta = 0.25
        super().__init__()

        self.latent = None
        self.q_latent = None

        # map the latent space to the codebook
        self.codebase = torch.randn(num_codes, latent_chan)

        self.encoder_downsampling = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=hidden_chan,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_chan),
            torch.nn.Conv2d(
                in_channels=hidden_chan,
                out_channels=hidden_chan,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_chan),
        )
        self.encoder_residual = torch.nn.Sequential(
            ResidualBlock(hidden_chan, hidden_chan),
            ResidualBlock(hidden_chan, hidden_chan),
        )
        self.encoder_compression = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=hidden_chan,
                out_channels=latent_chan,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(latent_chan),
        )
        self.encoder_stack = torch.nn.Sequential(
            self.encoder_downsampling, self.encoder_residual, self.encoder_compression
        )

        self.decoder_decompression = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=latent_chan,
                out_channels=hidden_chan,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_chan),
        )
        self.decoder_residual = torch.nn.Sequential(
            ResidualBlock(hidden_chan, hidden_chan),
            ResidualBlock(hidden_chan, hidden_chan),
        )
        self.decoder_upsampling = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=hidden_chan,
                out_channels=hidden_chan,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_chan),
            torch.nn.ConvTranspose2d(
                in_channels=hidden_chan,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # torch.nn.Tanh()
        )

        self.decoder_stack = torch.nn.Sequential(
            self.decoder_decompression, self.decoder_residual, self.decoder_upsampling
        )

    def quantize(self, x: torch.Tensor):
        b, c, h, w = x.shape

        # x: (B, C, H, W) -> (B, H, W, C)
        # x: (B, 256, 32, 32 -> (B, 32, 32, 256)
        x = x.permute(0, 2, 3, 1)

        # x: (B, H, W, C) -> (B*H*W, C)
        # x: (B, 32, 32, 256) -> (B*32*32, 256)
        x = x.reshape(-1, x.shape[-1])

        # distance: (B*H*W, num_codes)
        distances = torch.cdist(x, self.codebase)

        # get the closest code
        codes = torch.argmin(distances, dim=1)

        # reshape the codes to (B, H, W)
        codes = codes.reshape(b, h, w)

        maped_codes = self.codebase[codes]

        return codes, maped_codes

    def forward(self, x: torch.Tensor):
        self.latent = self.encoder_stack(x)

        # aka codes, discrete_latent: (B, H, W)
        discrete_latent, q_latent = self.quantize(self.latent.detach())

        # reshape to (B, C, H, W)
        q_latent = q_latent.reshape(self.latent.shape)

        # detach non differiable operation (quantization)
        # just copies the gradient from q_latent to latent
        self.q_latent = self.latent + (q_latent - self.latent).detach()

        reconstruction = self.decoder_stack(self.q_latent)

        return reconstruction

    def loss_fn(self, model_outputs, target):

        reconstruction_loss = torch.nn.functional.mse_loss(model_outputs, target)
        embed_loss = torch.nn.functional.mse_loss(self.latent.detach(), self.q_latent)
        commit_loss = torch.nn.functional.mse_loss(self.latent, self.q_latent.detach())

        loss = reconstruction_loss + embed_loss + self.beta * commit_loss

        self.batch_stats = {
            "Loss": loss.item(),
            "Reconstruction Loss": reconstruction_loss.item(),
            "Embed Loss": embed_loss.item(),
            "Commit Loss": commit_loss.item(),
        }

        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "VQ-VAE"
