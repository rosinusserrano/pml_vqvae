"""Utility classes / functions shared between different neural network modules."""

import torch
from torch import nn
import numpy as np
from scipy.stats import qmc


def upsample(in_channels: int, out_channels: int, activation: nn.Module = nn.ReLU()):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        activation,
        nn.BatchNorm2d(out_channels),
    )


def downsample(in_channels: int, out_channels: int, activation: nn.Module = nn.ReLU()):
    return nn.Sequential(
        # Downsampling
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        activation,
        nn.BatchNorm2d(out_channels),
    )


class PrintModule(nn.Module):
    """Simply prints the shape of the input.

    This "layer" doesn't really do anything and is just used for debugging. It
    simply prints out the shape of the input and can thus be used to examine
    how the data is transformed throughout the network."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class ResidualBlock(nn.Module):
    """Create residual layers."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        """Create a residual block."""
        super().__init__()

        self.downsample = in_channels != out_channels

        if kernel_size % 2 == 0:
            raise ValueError("Use odd kernel size for residual block.")

        padding = int((kernel_size - 1) / 2)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, affine=False),
        )
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection."""
        out = self.conv_block(x)

        if self.downsample:
            skip = self.skip_conv(x)
            out = out + skip
        else:
            out = self.skip_conv(out) + x

        return out


def sobol_uniform_points(n, d):
    """
    Generate n points uniformly distributed in d-dimensional space using Sobol sequence.

    Parameters:
    - n: int, the number of points
    - d: int, the number of dimensions

    Returns:
    - points: np.ndarray of shape (n, d), the uniformly distributed points
    """
    sampler = qmc.Sobol(d, scramble=True)
    points = sampler.random(n) * 2 - 1
    return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    p = np.random.uniform(-1, 1, (512, 2))
    plt.scatter(p[:, 0], p[:, 1], color="red", alpha=0.5, marker="o")
    plt.title("Uniform", fontsize=20)

    plt.subplot(1, 2, 2)
    p = sobol_uniform_points(512, 2)
    plt.scatter(p[:, 0], p[:, 1], color="red", alpha=0.5, marker="o")
    plt.title("Sobol", fontsize=20)

    plt.savefig("sobol.svg")
