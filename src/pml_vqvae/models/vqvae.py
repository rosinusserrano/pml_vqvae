"""Implementation of VQVAE"""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd

from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.nnutils import downsample, upsample, ResidualBlock


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
        batch = batch.reshape(-1, channels)  # flatten except for channels

        # Shape -> (batch_size * height * width)  x codebook_size
        squared_distances = torch.cdist(batch, codebook)

        closest_codes_indexes = torch.argmin(squared_distances, dim=1)

        # Save indexes in order to match gradients to corresponding codes
        ctx.save_for_backward(closest_codes_indexes, codebook)

        output = codebook[closest_codes_indexes]
        output = output.reshape(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        closest_codes_indexes = closest_codes_indexes.reshape(batch_size, height, width)

        return output, closest_codes_indexes

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        code_indexes, codebook = ctx.saved_tensors

        grad_encoder = grad_output

        n_channels = grad_output.shape[1]

        grad_output = grad_output.permute(0, 2, 3, 1)
        grad_output = grad_output.reshape(-1, n_channels)

        grad_codes = torch.zeros_like(codebook)
        grad_codes = torch.index_add(
            input=grad_codes,
            dim=0,
            index=code_indexes,
            source=grad_output,
        )

        return grad_encoder, grad_codes


@dataclass
class VQVAEConfig:
    "Config for VQVAE"
    name: str = "VQVAE"
    codebook_size: int
    commitment_weight: float
    hidden_dimension: int
    embedding_dimension: int


class VQVAE(PML_model):
    "Class for the VQVAE model"

    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            downsample(3, config.hidden_dimension),
            downsample(config.hidden_dimension, config.hidden_dimension),
            ResidualBlock(config.hidden_dimension, config.hidden_dimension),
            ResidualBlock(config.hidden_dimension, config.embedding_dimension),
        )

        self.codebook = nn.Parameter(
            torch.zeros(
                (config.codebook_size, config.embedding_dimension)
            ).data.uniform_(
                -1 / self.config.codebook_size,
                1 / self.config.codebook_size,
            ),
            requires_grad=True,
        )

        self.decoder = nn.Sequential(
            ResidualBlock(config.embedding_dimension, config.hidden_dimension),
            ResidualBlock(config.hidden_dimension, config.hidden_dimension),
            upsample(config.hidden_dimension, config.hidden_dimension),
            upsample(config.hidden_dimension, 3, activation=nn.Tanh()),
        )

    def forward(self, tensor: torch.Tensor):
        encoder_out = self.encoder(tensor)
        codes, indexes = VectorQuantization.apply(encoder_out, self.codebook)
        reconstruction = self.decoder(codes)

        return reconstruction, encoder_out, codes, indexes

    def loss_fn(self, model_outputs: torch.Tensor, target: torch.Tensor):
        reconstruction, encoder_out, codes, _ = model_outputs

        reconstruction = F.mse_loss(reconstruction, target)

        encoder_commitment = F.mse_loss(codes.detach(), encoder_out)
        encoder_commitment *= self.config.commitment_weight

        codes_commitment = F.mse_loss(codes, encoder_out.detach())

        loss = reconstruction + encoder_commitment + codes_commitment

        self.batch_stats = {
            "Loss": loss.item(),
            "Reconstruction": reconstruction.item(),
            "Commitment (encoder)": encoder_commitment.item(),
            "Commitment (codes)": codes_commitment.item(),
        }

        return loss

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def name(self):
        return "VQVAE"

    def visualize_output(self, output):
        return output[0]
