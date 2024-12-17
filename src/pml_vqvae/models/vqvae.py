"""Implementation of VQVAE"""

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd

from pml_vqvae.models.baseline.pml_model_interface import PML_model
from pml_vqvae.nnutils import downsample, upsample, zip_channels_list, ResidualBlock


class VectorQuantization(autograd.Function):
    """Function to perform vector quantization and copy the codebook gradients
    to the encoders output"""

    @staticmethod
    def forward(ctx, batch, codebook):
        batch_size, channels, height, width = batch.shape
        codebook_size, embedding_dim = codebook.shape

        if embedding_dim != channels:
            raise ValueError("codebook embedding dimension doesnt equal" "channel dim!")

        batch = batch.permute(0, 2, 3, 1)  # channels on last dim
        batch = batch.view(-1, channels)  # flatten except for channels

        print("Codebook shape for broadcasting: ", codebook.shape)
        print("Batch shape for broadcasting: ", batch.shape)

        # Shape -> (batch_size * height * width)  x codebook_size
        # TODO: use ||x - z||² == ||x||² + ||z||² - 2 * x.T @ z
        squared_distances = torch.sum(
            (batch[:, None, :] - codebook[None, :, :]) ** 2, dim=-1
        )

        closest_codes_indexes = torch.argmin(squared_distances, dim=1)

        # Save indexes in order to match gradients to corresponding codes
        ctx.save_for_backward(closest_codes_indexes)

        output = codebook[closest_codes_indexes]
        output = output.view(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # code_indexes, = ctx.saved_tensors

        # n_channels = grad_output.shape[1]

        # grad_output = grad_output.permute(0, 2, 3, 1)
        # grad_output = grad_output.view(-1, n_channels)
        return grad_output, None


@dataclass
class VQVAEConfig:
    "Config for VQVAE"
    name: str = "VQVAE"
    codebook_size: int = 512
    commitment_weight: float = 0.25
    downsampling_channels: list[int] = field(default_factory=lambda: [3, 256, 256])
    encoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256]
    )
    compression_channels: int = 16
    decoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256]
    )
    upsampling_channels: list[int] = field(default_factory=lambda: [256, 256, 3])


class VQVAE(PML_model):
    "Class for the VQVAE model"

    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            *[
                downsample(inc, outc)
                for inc, outc in zip_channels_list(config.downsampling_channels)
            ],
            *[
                ResidualBlock(inc, outc)
                for inc, outc in zip_channels_list(config.encoder_residual_channels)
            ],
            nn.Conv2d(  # compression
                in_channels=config.encoder_residual_channels[-1],
                out_channels=config.compression_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(config.compression_channels)
        )

        self.codebook = nn.Parameter(
            torch.randn((config.codebook_size, config.compression_channels))
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(  # decompression
                in_channels=config.compression_channels,
                out_channels=config.decoder_residual_channels[0],
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(config.decoder_residual_channels[0]),
            *[
                ResidualBlock(inc, outc)
                for inc, outc in zip_channels_list(config.decoder_residual_channels)
            ],
            *[
                upsample(inc, outc)
                for inc, outc in zip_channels_list(config.upsampling_channels[:-1])
            ],
            upsample(
                config.upsampling_channels[-2],
                config.upsampling_channels[-1],
                activation=nn.Tanh(),
            )
        )

    def forward(self, tensor: torch.Tensor):
        encoder_out = self.encoder(tensor)
        codes = VectorQuantization.apply(encoder_out, self.codebook)
        reconstruction = self.decoder(codes)

        if self.training:
            return reconstruction, encoder_out, codes

        return reconstruction

    def loss_fn(self, model_outputs: torch.Tensor, target: torch.Tensor):
        reconstruction, encoder_out, codes = model_outputs
        return (
            F.mse_loss(reconstruction, target)
            + ((codes.detach() - encoder_out) ** 2).sum()
            + (
                self.config.commitment_weight
                * ((encoder_out.detach() - codes) ** 2).sum()
            )
        )

    def backward(self, loss: torch.Tensor):
        loss.backward()

    @staticmethod
    def collect_stats(output, target, loss):
        return {"Loss": loss.item()}

    def name(self):
        return "VQVAE"

    @staticmethod
    def visualize_output(batch, output, target, prefix="", base_dir="."):
        return None


if __name__ == "__main__":
    model_conf = VQVAEConfig()
    model = VQVAE(model_conf)
    img = torch.randn((1, 3, 32, 32))
    print([out.shape for out in model(img)])
