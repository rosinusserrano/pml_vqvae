"""Utility classes / functions shared between different neural network modules."""

from argparse import ArgumentError
import torch
from torch import nn


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
            nn.BatchNorm2d(out_channels),
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
