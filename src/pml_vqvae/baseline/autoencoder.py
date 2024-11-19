import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show_image_grid
from pml_vqvae.nnutils import ResidualBlock


# Ich habe das auskommentiert und meinen residual block genommen, weil ich
# glaube hier ist ein fehler dass hier ein fehler drin ist. Die 1x1
# convolution glaube ich müsste parallel gemacht werden um für eventuelle
# Änderungen der anzahl der channels zu kompensieren. Habe es trotzdem erstmal
# drin gelasse in case, dass ich es falsch verstanden habe.

# class ResidualBlock(torch.nn.Module):
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(
#                 in_channels=in_chan,
#                 out_channels=out_chan,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(
#                 in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1
#             ),
#         )

#     def forward(self, x):
#         return torch.nn.functional.relu(x + self.conv(x))


class BaselineAutoencoder(PML_model):
    def __init__(self):
        hidden_chan = 128
        latent_chan = 6
        super().__init__()

        self.encoder_downsampling = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=hidden_chan, kernel_size=4, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_chan,
                out_channels=hidden_chan,
                kernel_size=4,
                stride=2,
            ),
            torch.nn.ReLU(),
        )
        self.encoder_residual = torch.nn.Sequential(
            ResidualBlock(hidden_chan, hidden_chan),
            ResidualBlock(hidden_chan, hidden_chan),
        )
        self.encoder_compression = torch.nn.Conv2d(
            in_channels=hidden_chan, out_channels=latent_chan, kernel_size=1, stride=1
        )
        self.encoder_stack = torch.nn.Sequential(
            self.encoder_downsampling, self.encoder_residual, self.encoder_compression
        )

        self.decoder_decompression = torch.nn.Conv2d(
            in_channels=latent_chan, out_channels=hidden_chan, kernel_size=1, stride=1
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
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=hidden_chan, out_channels=3, kernel_size=4, stride=2
            ),
            torch.nn.ReLU(),
        )

        self.decoder_stack = torch.nn.Sequential(
            self.decoder_decompression, self.decoder_residual, self.decoder_upsampling
        )

    def forward(self, x):
        latent = self.encoder_stack(x)
        reconstruction = self.decoder_stack(latent)

        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)
        return reconstruction

    @staticmethod
    def loss_fn(model_outputs, target):
        return torch.nn.functional.mse_loss(model_outputs, target)

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def collect_stats(output, target, loss):
        return {"Loss": loss.item()}

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show_image_grid(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show_image_grid(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "BaselineAutoencoder"


def simple_downsample(image, scale):
    new_img = np.zeros((128, 128, 3), dtype=np.float32)
    for i in range(0, 128):
        for j in range(0, 128):
            new_img[i][j] = image[math.floor(i * scale)][math.floor(j * scale)] / 255.0
    return new_img


def main():
    plt.figure(figsize=(3, 3))

    img_array = np.array(Image.open("data/hand.webp"))
    print(img_array.shape)
    img_array = simple_downsample(img_array, 2.5)
    plt.imshow(img_array)
    plt.show()

    img_array = np.array(Image.open("data/hand.webp"))
    print(img_array.shape)
    img_array = simple_downsample(img_array, 2.5)
    """
    img_tensor = torch.tensor(img_array, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)
    model = BaselineAutoencoder()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i in range(0, 10001):
        optimizer.zero_grad()
        decoded = model.forward(img_tensor)
        loss = loss_fn(img_tensor, decoded)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:

            decoded_img = decoded.detach().squeeze().permute(1, 2, 0).numpy()

            plt.figure(figsize=(3, 3))
            plt.title(f'after {i}')
            plt.imshow(decoded_img)
            plt.show()

"""


if __name__ == "__main__":
    main()
