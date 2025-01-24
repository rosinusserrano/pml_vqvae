"""Implementation of VQVAE"""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd

from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.nnutils import downsample, upsample, ResidualBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        nearest_codes_indexes = torch.argmin(squared_distances, dim=1)

        # Save indexes in order to match gradients to corresponding codes
        ctx.save_for_backward(nearest_codes_indexes, codebook)

        output = codebook[nearest_codes_indexes]
        output = output.reshape(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        nearest_codes_indexes = nearest_codes_indexes.reshape(batch_size, height, width)

        return output, nearest_codes_indexes

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
    codebook_size: int
    commitment_weight: float
    hidden_dimension: int
    embedding_dimension: int
    codebook_initialization_radius: float
    name: str = "VQVAE"


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
                -config.codebook_initialization_radius,
                config.codebook_initialization_radius,
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

    @torch.no_grad()
    def encode(self, tensor: torch.Tensor):
        encoder_out = self.encoder(tensor)
        _, indexes = VectorQuantization.apply(encoder_out, self.codebook)

        return indexes

    def decode(self, indexes: torch.Tensor):
        indexes = indexes.squeeze()
        bs, h, w = indexes.shape
        indexes = indexes.flatten()

        codes = self.codebook[indexes]

        codes = codes.reshape(bs, h, w, -1)
        codes = codes.permute(0, 3, 1, 2)

        return self.decoder(codes)

    def loss_fn(self, model_outputs: torch.Tensor, target: torch.Tensor):
        reconstruction, encoder_out, codes, indexes = model_outputs

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
            "Code usage": set(indexes.flatten().tolist()),
        }

        return loss

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def name(self):
        return "VQVAE"

    def visualize_output(self, output):
        return output[0]


class VectorQuantizationWithCdist(VectorQuantization):
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

        nearest_codes_indexes = torch.argmin(squared_distances, dim=1)

        # Save indexes in order to match gradients to corresponding codes
        ctx.save_for_backward(nearest_codes_indexes, codebook)

        output = codebook[nearest_codes_indexes]
        output = output.reshape(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        nearest_codes_indexes = nearest_codes_indexes.reshape(batch_size, height, width)

        return output, nearest_codes_indexes, squared_distances

    @staticmethod
    def backward(ctx, grad_output, grad_indices, grad_return_cdist=None):
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

        return grad_encoder, grad_codes, None


@dataclass
class VQVAECodeEnforcedConfig(VQVAEConfig):
    buffer_size: int = 20
    max_idle_weight: int = 3


class VQVAECodeEnforced(VQVAE):
    """This variant adds a loss term that draws unused codes towards the mean
    of the encoder output.

    This loss is scaled for each code individually by the "idle count". The
    more iterations a given code is left unused, the stronger it is drawn
    towards the encoders output mean.
    """

    def __init__(self, config: VQVAECodeEnforcedConfig):
        super().__init__(config)
        self.config = config

        self.code_idle_count = torch.zeros(
            (self.config.codebook_size,),
            dtype=torch.long,
        ).to(DEVICE)

    def forward(self, tensor: torch.Tensor):
        encoder_out = self.encoder(tensor)
        codes, indexes, cdist = VectorQuantizationWithCdist.apply(
            encoder_out,
            self.codebook,
        )
        reconstruction = self.decoder(codes)

        nearest_encoder_indexes = torch.argmin(cdist, dim=0)
        nearest_encoder_embeds = encoder_out.permute(0, 2, 3, 1).reshape(
            -1, self.config.embedding_dimension
        )[nearest_encoder_indexes]

        return reconstruction, encoder_out, codes, indexes, nearest_encoder_embeds

    def loss_fn(self, model_outputs, target):
        reconstruction, encoder_out, codes, indices, nearest_encoder_embeds = (
            model_outputs
        )

        # Reconstructoin loss
        reconstruction = F.mse_loss(reconstruction, target)

        # Commitment losses
        encoder_commitment = F.mse_loss(codes.detach(), encoder_out)
        encoder_commitment *= self.config.commitment_weight

        codes_commitment = F.mse_loss(codes, encoder_out.detach())

        # Code enforcment loss
        unique_indices = indices.unique()
        self.code_idle_count += 1
        self.code_idle_count[unique_indices] = -self.config.buffer_size

        n_unused_codes = self.config.codebook_size - unique_indices.shape[0]
        code_enforcement = torch.sum(
            (torch.sum((self.codebook - nearest_encoder_embeds) ** 2, dim=1))
            * torch.clamp(self.code_idle_count, 0, self.config.max_idle_weight)
        ) / (n_unused_codes if n_unused_codes > 0 else 1)

        loss = reconstruction + encoder_commitment + codes_commitment + code_enforcement

        self.batch_stats = {
            "Loss": loss.item(),
            "Reconstruction": reconstruction.item(),
            "Commitment (encoder)": encoder_commitment.item(),
            "Commitment (codes)": codes_commitment.item(),
            "Code usage": set(indices.flatten().tolist()),
            "Enforcement": code_enforcement.item(),
            "Max idle count": torch.max(self.code_idle_count).item(),
            "Num idle codes": torch.sum(self.code_idle_count > 0).item(),
        }

        return loss

    def name(self):
        return "VQVAECodeEnforced"


class VectorQuantizationMeanCodeGradient(autograd.Function):
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

        nearest_codes_indexes = torch.argmin(squared_distances, dim=1)

        # Save indexes in order to match gradients to corresponding codes
        ctx.save_for_backward(nearest_codes_indexes, codebook)

        output = codebook[nearest_codes_indexes]
        output = output.reshape(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        nearest_codes_indexes = nearest_codes_indexes.reshape(batch_size, height, width)

        return output, nearest_codes_indexes

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
        idx_count = torch.bincount(code_indexes, minlength=codebook.shape[0])
        idx_count[idx_count == 0] = 1
        grad_codes = grad_codes / idx_count[:, None]

        return grad_encoder, grad_codes
