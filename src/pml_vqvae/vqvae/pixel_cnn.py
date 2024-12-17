import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from torch import nn
from torch.nn import functional as F

from PIL import Image
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show
from pml_vqvae.nnutils import ResidualBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MaskedConv2d(nn.Module):

    def __init__(
        self,
        mask: torch.Tensor,
        in_channels: int,
        out_channels: int,
        dilation: int = 1
    ):
        super().__init__()

        self.mask = mask.to(DEVICE)

        padding_0 = (mask.shape[0] - 1) // 2 * dilation
        padding_1 = (mask.shape[1] - 1) // 2 * dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=mask.shape,
            padding=(padding_0, padding_1),
            dilation=dilation
        )

    def forward(self, tensor: torch.Tensor):
        self.conv.weight.data *= self.mask[None, None, :]
        return self.conv(tensor)


class VerticalMaskedConvolution(MaskedConv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        first_layer: bool = False,
        dilation: int = 1,
    ):
        mask = torch.ones((kernel_size, kernel_size))
        mask[(kernel_size // 2) + 1 :, :] = 0

        if first_layer:
            mask[kernel_size // 2, :] = 0

        super().__init__(
            mask=mask,
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
        )


class HorizontalMaskedConvolution(MaskedConv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        first_layer: bool = False,
        dilation: int = 1,
    ):
        mask = torch.ones((1, kernel_size))
        mask[:, (kernel_size // 2) + 1 :] = 0

        if first_layer:
            mask[:, kernel_size // 2] = 0

        super().__init__(
            mask=mask,
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
        )


class GatedMaskedConvolution(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.vconv = VerticalMaskedConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.hconv = HorizontalMaskedConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.vconv_1x1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        self.hconv_1x1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, vertical_and_horizontal: tuple[torch.Tensor, torch.Tensor]):
        vertical, horizontal = vertical_and_horizontal

        vertical = self.vconv(vertical)

        vertical_out = F.tanh(vertical[:, : self.hidden_dim]) * F.sigmoid(
            vertical[:, self.hidden_dim :]
        )

        horizontal_out = self.hconv(horizontal)
        horizontal_out += self.vconv_1x1(F.leaky_relu(vertical))
        horizontal_out = F.tanh(horizontal_out[:, : self.hidden_dim]) * F.sigmoid(
            horizontal_out[:, self.hidden_dim :]
        )
        horizontal_out = self.hconv_1x1(horizontal_out)
        horizontal_out = F.leaky_relu(horizontal_out) + horizontal

        return (vertical_out, horizontal_out)


class ConditionalGatedMaskedConvolution(GatedMaskedConvolution):

    def __init__(self, num_embeddings, hidden_dim, kernel_size, dilation: int = 1):
        super().__init__(hidden_dim, kernel_size, dilation)

        self.embeddings = nn.Embedding(num_embeddings, hidden_dim)

    def forward(self, vertical_and_horizontal_and_labels):
        vertical, horizontal, labels = vertical_and_horizontal_and_labels
        conditionals = self.embeddings(labels)

        vertical = vertical + conditionals[:, :, None, None]

        return *super().forward((vertical, horizontal)), labels


class GatedPixelCNN(PML_model):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_quantization: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(
                "Only works for 1 channel, e.g. grayscale images or latent codes of VQ-VAE."
            )

        self.n_quantization = n_quantization

        self.input_vconv = VerticalMaskedConvolution(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            first_layer=True,
            dilation=dilation,
        )

        self.input_hconv = HorizontalMaskedConvolution(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            first_layer=True,
            dilation=dilation,
        )

        self.hidden_layers = nn.Sequential(
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
            GatedMaskedConvolution(hidden_dim, kernel_size, dilation),
        )

        self.output_conv = nn.Conv2d(hidden_dim, n_quantization, kernel_size=1)

    def forward(self, tensor: torch.Tensor):
        tensor = tensor * 2 - 1  # this has to go after testing!!!!

        vertical = F.leaky_relu(self.input_vconv(tensor))
        horizontal = F.leaky_relu(self.input_hconv(tensor))

        _, horizontal_out = self.hidden_layers((vertical, horizontal))

        return self.output_conv(F.leaky_relu(horizontal_out))

    def loss_fn(self, model_outputs, target):
        n_quantization = model_outputs.shape[1]

        model_outputs = model_outputs.permute(0, 2, 3, 1).reshape((-1, n_quantization))
        target = (target.permute(0, 2, 3, 1).reshape(-1, 1) * 255).long()

        loss = F.cross_entropy(model_outputs, target)

        self.batch_stats = {"loss": loss.item()}
        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))

        output_images = torch.argmax(output, dim=1).long()
        show(
            output_images,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )
    
    @torch.no_grad()
    def generate_images(self):
        self.eval()

        batch_size = 8
        height = width = 28
        generated_image = torch.zeros((batch_size, 1, height, width)).to(DEVICE)
        for h in range(height):
            for w in range(width):
                preds = self.forward(generated_image)

                # preds = torch.argmax(preds, dim=1, keepdim=True)
                # preds = preds[:, :, h, w]
                # preds = preds / 255

                distributions = preds[:, :, h, w].detach().cpu()
                distributions = F.softmax(distributions, dim=-1)

                vals = torch.multinomial(distributions, num_samples=1) / 255

                generated_image[:, :, h, w] = vals.to(DEVICE)
        
        self.train()

        return generated_image

    def name(self):
        return "PixelCNN"


class ConditionalPixelCNN(PML_model):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_quantization: int,
        n_classes: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(
                "Only works for 1 channel, e.g. grayscale images or latent codes of VQ-VAE."
            )

        self.n_quantization = n_quantization
        self.n_classes = n_classes

        self.input_vconv = VerticalMaskedConvolution(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            first_layer=True,
            dilation=1,
        )

        self.input_hconv = HorizontalMaskedConvolution(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            first_layer=True,
            dilation=1,
        )

        self.hidden_layers = nn.Sequential(
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=1),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=2),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=1),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=3),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=1),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=4),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=1),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=3),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=1),
            ConditionalGatedMaskedConvolution(n_classes, hidden_dim, kernel_size, dilation=4),
        )

        self.output_conv = nn.Conv2d(hidden_dim, n_quantization, kernel_size=1)

    def forward(self, tensor: torch.Tensor, class_idx: torch.Tensor):
        tensor = tensor * 2 - 1  # this has to go after testing!!!!

        vertical = F.leaky_relu(self.input_vconv(tensor))
        horizontal = F.leaky_relu(self.input_hconv(tensor))

        _, horizontal_out, _ = self.hidden_layers((vertical, horizontal, class_idx))

        return self.output_conv(F.leaky_relu(horizontal_out))

    def loss_fn(self, model_outputs, target):
        n_quantization = model_outputs.shape[1]

        loss = F.cross_entropy(model_outputs, (target.squeeze() * 255).long())

        self.batch_stats = {"loss": loss.item()}
        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))

        output_images = torch.argmax(output, dim=1).long()
        show(
            output_images,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )
    
    @torch.no_grad()
    def generate_images(self):
        self.eval()

        batch_size = 8
        height = width = 28
        generated_image = torch.zeros((batch_size, 1, height, width)).to(DEVICE)
        labels = torch.arange(0, batch_size).long().to(DEVICE)
        for h in range(height):
            for w in range(width):
                preds = self.forward(generated_image, labels)

                distributions = preds[:, :, h, w].detach().cpu()
                distributions = F.softmax(distributions, dim=-1)

                vals = torch.multinomial(distributions, num_samples=1) / 255

                generated_image[:, :, h, w] = vals.to(DEVICE)
        
        self.train()

        return generated_image

    def name(self):
        return "PixelCNN"
