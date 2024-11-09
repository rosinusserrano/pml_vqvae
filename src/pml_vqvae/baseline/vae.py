"""Simple implementation of VAE.

very simple
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm

from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show_image_grid
from pml_vqvae.dataset.dataloader import load_data


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


def encoder() -> nn.Sequential:
    """Create basic encoder."""
    return nn.Sequential(
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


def reparameterization_trick(
    encoder_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the reparameteriztion trick.

    Splits the output of the encoder into logvar and mean and then samples
    using the reparameterization trick.
    """
    batch_size, n_feats, height, width = encoder_output.shape

    assert n_feats % 2 == 0, """Use even number of output features otherwise
    one can't split them into mean and variance"""  # noqa: S101

    z = torch.randn((batch_size, n_feats // 2, height, width))
    mean = encoder_output[:, : (n_feats // 2), ...]
    logvar = encoder_output[:, (n_feats // 2) :, ...]
    z_hat = mean + torch.exp(0.5 * logvar) * z

    return z_hat, mean, logvar


def decoder() -> nn.Sequential:
    """Return decoder."""
    return nn.Sequential(
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


def loss_function(
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Compute the VAE loss.

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

    return {
        "loss": loss,
        "Reconstruction_Loss": reconstruction_loss.detach(),
        "KLD": -kld_loss.detach(),
    }


def overfit_on_first_batch():
    "In order to check if your model works as wished, test if it can overfit."
    train_dl, _ = load_data("cifar", batch_size=4, n_train=10, shuffle=True, seed=2024)

    n_epochs = 100000

    vae_co = encoder()
    vae_dec = decoder()

    optimizer = torch.optim.Adam([*vae_co.parameters(), *vae_dec.parameters()])

    reconstruction_loss = []
    kld_loss = []

    x = None
    for batch, _ in train_dl:
        x = batch
        break

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        z = vae_co(x)
        z_hat, mean, logvar = reparameterization_trick(z)
        x_hat = vae_dec(z_hat)

        loss = loss_function(x, mean, logvar, x_hat)
        loss["loss"].backward()

        kld_loss.append(loss["KLD"].item())
        reconstruction_loss.append(loss["Reconstruction_Loss"].item())

        optimizer.step()

        if epoch % 100 == 0:
            print(f"EPOCH {epoch}")

        if epoch % 100 == 0:
            show_image_grid(x, outfile="test.output/orig.png")
            show_image_grid(x_hat, outfile="test.output/recon.png")

            plt.clf()
            plt.plot(kld_loss)
            plt.plot(reconstruction_loss)
            plt.savefig("test.output/losses.png")


def train_for_real():
    train_dl, _ = load_data("cifar", batch_size=64, shuffle=True, seed=2024)

    n_epochs = 100000

    vae_co = encoder()
    vae_dec = decoder()

    optimizer = torch.optim.Adam([*vae_co.parameters(), *vae_dec.parameters()])

    reconstruction_loss = []
    kld_loss = []

    for epoch in range(n_epochs):
        for x, _ in tqdm(train_dl):
            optimizer.zero_grad()

            z = vae_co(x)
            z_hat, mean, logvar = reparameterization_trick(z)
            x_hat = vae_dec(z_hat)

            loss = loss_function(x, mean, logvar, x_hat)
            loss["loss"].backward()

            kld_loss.append(loss["KLD"].item())
            reconstruction_loss.append(loss["Reconstruction_Loss"].item())

            optimizer.step()

        if epoch % 10 == 0:
            print(f"EPOCH {epoch}")
            print(f"loss {loss['loss'].item()}")
            print(f"kld loss {kld_loss[-1]}")
            print(f"recon loss {reconstruction_loss[-1]}")

        if epoch % 100 == 0:
            show_image_grid(x, outfile="test.output/orig.png")
            show_image_grid(x_hat, outfile="test.output/recon.png")

            z_samesize = torch.randn((4, 8, 4, 4))
            z_diffsize = torch.randn((4, 8, 6, 6))
            x_samesize = vae_dec(z_samesize)
            x_diffsize = vae_dec(z_diffsize)
            show_image_grid(x_samesize, outfile="test.output/samplesame.png")
            show_image_grid(x_diffsize, outfile="test.output/samplediff.png")

            plt.clf()
            plt.plot(kld_loss)
            plt.plot(reconstruction_loss)
            plt.savefig("test.output/losses.png")
