import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show
from pml_vqvae.nnutils import ResidualBlock


class BaselineAutoencoder(PML_model):
    def __init__(self):
        hidden_chan = 128
        latent_chan = 2
        super().__init__()

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
        )

        self.decoder_stack = torch.nn.Sequential(
            self.decoder_decompression, self.decoder_residual, self.decoder_upsampling
        )

    def forward(self, x):
        latent = self.encoder_stack(x)
        reconstruction = self.decoder_stack(latent)

        return reconstruction

    @staticmethod
    def loss_fn(model_outputs, target):
        return torch.nn.functional.mse_loss(model_outputs, target)

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def collect_stats(output, target, loss):
        return {"Loss": loss.item()}

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "BaselineAutoencoder"
