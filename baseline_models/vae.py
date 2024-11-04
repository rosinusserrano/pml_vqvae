"Baseline models to compare VQ-VAE to"

import torch
from torch import nn

from dataloaders import load_cifar10


def conv_block(inc, outc):
    """A simple 3-layered conv block with kernel size 3, padding and relu
    activation"""
    return nn.Sequential(nn.Conv2d(inc, outc, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outc, outc, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outc, outc, 3, padding=1), nn.ReLU())


class ResidualBlock(nn.Module):
    "Class to create residual layers"

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = conv_block(in_channels, out_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        """Feeds input (a) through the conv block and (b) through a 1x1
        convolutional layer that adjust its channels so that it can be added
        with the output of the conv block."""
        out = self.conv_block(x)
        skip = self.skip_conv(x)
        return out + skip


def encoder():
    "The encoder for the simple VAE"
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
        nn.Conv2d(64, 2, 1))


def reparamterization_trick(encoder_output: torch.Tensor):
    """Splits the output of the encoder into logvar and mean and then samples
    using the reparameterization trick"""
    batch_size, height, width, n_feats = encoder_output.shape

    assert n_feats % 2 == 0, """Use even number of output features otherwise
    one can't split them into mean and variance"""

    z = torch.randn((batch_size, height, width, n_feats // 2))
    mean = encoder_output[..., :(n_feats // 2)]
    logvar = encoder_output[..., (n_feats // 2):]
    z_hat = mean + torch.exp(0.5 * logvar) * z

    return z_hat, mean, logvar


def decoder():
    "The decoder for the simple VAE"
    return nn.Sequential(
        # 28x28x1 -> 32x32x64
        nn.ConvTranspose2d(1, 64, 5),
        nn.ReLU(),
        # 32x32x64 -> 32x32x16
        ResidualBlock(64, 16),
        # 32x32x16 -> 64x64x16
        nn.ConvTranspose2d(16, 16, 3, 2, 1),
        nn.ReLU(),
        # 64x64x16 -> 64x64x8
        ResidualBlock(16, 8),
        # 64x64x8 -> 128x128x8
        nn.ConvTranspose2d(8, 8, 3, 2, 1),
        nn.ReLU(),
        # 128x128x8 -> 128x128x3
        ResidualBlock(8, 3))


def loss_function(original, mean, logvar, reconstruction):
    """Computes the VAE loss which consist of reconstruction loss and KL
    divergence between prior distribution z ~ N(0, I) and posterior 
    distribution z|x ~ N(mean, exp(logvar))"""

    reconstruction_loss = torch.mean((original - reconstruction)**2)

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=(1, 2, 3)),
        dim=0)

    loss = reconstruction_loss + kld_loss

    return {
        'loss': loss,
        'Reconstruction_Loss': reconstruction_loss.detach(),
        'KLD': -kld_loss.detach()
    }


if __name__ == "__main__":
    train_dl, test_dl = load_cifar10(batch_size=32)

    n_epochs = 100

    for epoch in n_epochs:
        for batch, _ in train_dl:
            
