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


class PixelCNN(PML_model):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return

    def loss_fn(self, model_outputs, target):
        return F.mse_loss(model_outputs, target)

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "PixelCNN"


class MaskedConv2d(nn.Module):

    def __init__(
        self,
        mask: torch.Tensor,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.mask = mask
        self.register_buffer(
            "conv2dmask",
            mask,
        )  # if it doesn't work, write mask[None, None]

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=mask.shape,
            padding=((mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2),
        )

    def forward(self, tensor: torch.Tensor):
        self.conv.weight.data *= self.mask
        return self.conv(tensor)


class VerticalMaskedConvolution(MaskedConv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        first_layer: bool = False,
    ):
        mask = torch.ones((kernel_size, kernel_size))
        mask[(kernel_size // 2) + 1 :, :] = 0

        if first_layer:
            mask[kernel_size // 2, :] = 0

        super().__init__(
            mask=mask,
            in_channels=in_channels,
            out_channels=out_channels,
        )


class HorizontalMaskedConvolution(MaskedConv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        first_layer: bool = False,
    ):
        mask = torch.ones((1, kernel_size))
        mask[:, (kernel_size // 2) + 1 :] = 0

        if first_layer:
            mask[:, kernel_size // 2] = 0

        super().__init__(
            mask=mask,
            in_channels=in_channels,
            out_channels=out_channels,
        )


class GatedMaskedConvolution(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.vconv = VerticalMaskedConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
        )

        self.hconv = HorizontalMaskedConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
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


class GatedPixelCNN(PML_model):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_quantization: int,
        kernel_size: int = 3,
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
        )

        self.input_hconv = HorizontalMaskedConvolution(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            first_layer=True,
        )

        self.hidden_layers = nn.Sequential(
            GatedMaskedConvolution(hidden_dim, kernel_size),
            GatedMaskedConvolution(hidden_dim, kernel_size),
            GatedMaskedConvolution(hidden_dim, kernel_size),
            GatedMaskedConvolution(hidden_dim, kernel_size),
            GatedMaskedConvolution(hidden_dim, kernel_size),
        )

        self.output_conv = nn.Conv2d(hidden_dim, n_quantization, kernel_size=1)

    def forward(self, tensor: torch.Tensor):
        tensor = tensor * 2 - 1  # this has to go after testing!!!!

        vertical = F.leaky_relu(self.input_vconv(tensor))
        horizontal = F.leaky_relu(self.input_hconv(tensor))

        _, horizontal_out = self.hidden_layers((vertical, horizontal))

        return self.output_conv(F.leaky_relu(horizontal_out))

    def loss_fn(self, model_outputs, target):
        return F.cross_entropy(model_outputs, target)

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

    def name(self):
        return "PixelCNN"


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    from torchvision.utils import make_grid

    ds = MNIST("data", download=True, transform=v2.ToTensor())
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    model = GatedPixelCNN(
        in_channels=1,
        hidden_dim=256,
        n_quantization=256,
        kernel_size=3,
    )

    optimizer = torch.optim.Adam(model.parameters())

    for index, (img, _) in enumerate(loader):
        optimizer.zero_grad()

        preds = model(img).permute(0, 2, 3, 1).reshape(-1, 256)
        img_long = (img * 255).long().permute(0, 2, 3, 1).reshape(-1)

        loss = F.cross_entropy(preds, img_long)
        loss.backward()

        print(f"Loss: {loss.item()} ({index}/{len(loader)})")

        optimizer.step()

        if index > 100:
            break

    generated_image = torch.zeros((1, 1, 28, 28))

    for h in range(28):
        for w in range(28):
            print(f"{w + (h * 28)} / {28 * 28}")

            pred = model(generated_image)
            pred = torch.argmax(pred, dim=1, keepdim=True)

            pred = pred[:, :, h, w]
            print(f"Filling first with {pred[0].flatten().item()}")
            pred = pred / 255

            generated_image[:, :, h, w] = pred

    grid = make_grid(generated_image, pad_value=1, padding=1)

    plt.imshow(generated_image[0].permute(1, 2, 0))
    plt.show()
