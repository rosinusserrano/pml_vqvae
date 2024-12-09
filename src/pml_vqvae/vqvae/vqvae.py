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
    """
    VQ-VAE model
    """

    def __init__(
        self, hidden_chan: int = 128, latent_chan: int = 128, num_codes: int = 512
    ):
        """
        VQ-VAE model
        :param hidden_chan: number of channels in the hidden layer
        :param latent_chan: number of channels in the latent layer
        :param num_codes: number of codes in the codebook
        """

        self.hidden_chan = hidden_chan
        self.latent_chan = latent_chan  # D in the paper
        self.num_codes = num_codes  # K in the paper
        self.beta = 0.25
        super().__init__()

        self.latent = None
        self.q_latent = None

        # codebook of shape (num_codes, latent_chan)
        self.codebase = torch.nn.Parameter(
            # torch.FloatTensor(num_codes, latent_chan, requires_grad=True).uniform_(-1, 1).to("cuda")
            # torch.linspace(-1, 1, num_codes)
            # .reshape(-1, 1)
            # .repeat(1, latent_chan)
            torch.torch.rand(num_codes, latent_chan),
            requires_grad=True,
        )

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
        self.codes = codes.reshape(b, h, w)

        # get code from codebase
        maped_codes = self.codebase[self.codes]

        return codes, maped_codes

    def forward(self, x: torch.Tensor):
        self.latent = self.encoder_stack(x)

        # print(self.latent.permute(0, 2, 3, 1)[0])

        # aka codes, discrete_latent: (B, H, W)
        self.discrete_latent, q_latent = self.quantize(self.latent.detach())

        # reshape to (B, C, H, W)
        q_latent = q_latent.permute(0, 3, 1, 2)

        # detach non differiable operation (quantization)
        # just copies the gradient from q_latent to latent
        self.q_latent = self.latent + (q_latent - self.latent).detach()

        reconstruction = self.decoder_stack(self.latent)

        return reconstruction

    def loss_fn(self, model_outputs, target):

        e = self.codebase[self.discrete_latent].reshape(self.latent.shape)

        reconstruction_loss = torch.nn.functional.mse_loss(model_outputs, target)
        embed_loss = torch.nn.functional.mse_loss(self.latent.detach(), e)
        commit_loss = torch.nn.functional.mse_loss(self.latent, e.detach())

        loss = reconstruction_loss + embed_loss + self.beta * commit_loss

        self.batch_stats = {
            "Loss": loss.detach().cpu().item(),
            "Reconstruction Loss": reconstruction_loss.detach().cpu().item(),
            "Embed Loss": embed_loss.detach().cpu().item(),
            "Commit Loss": commit_loss.detach().cpu().item(),
        }

        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    def vis_codes(self, idx, base_dir: str = "."):
        fig = plt.figure(figsize=(8, 8))
        colors = ["r", "g", "b", "y", "m", "c"]

        latent = self.latent.cpu().detach().numpy()[0]
        latent = np.moveaxis(latent, 0, -1)
        latent = latent.reshape(-1, 2)

        codes = self.codes.cpu().detach().numpy()[0].flatten()

        for i, c in enumerate(self.codebase):
            c = c.cpu().detach().numpy()
            plt.scatter(c[0], c[1], marker="o", color=colors[i], label=f"{c}")

        for la, c_dx in zip(latent, codes):
            plt.scatter(la[0], la[1], marker="x", color=colors[c_dx])

        plt.legend()

        # set window
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.title("Codes")
        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.savefig(os.path.join(base_dir, f"codes_{idx}.png"))
        plt.clf()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "VQ-VAE"
