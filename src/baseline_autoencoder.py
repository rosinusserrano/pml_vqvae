import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class BaselineAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_downsample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.encoder_resnet = torch.nn.Res

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=4, stride=2, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2)
            torch.nn.ReLU()

        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


img_array = np.array(Image.open('data/os361x7rfy151.webp'))[0:1024, 0:1024]
print(img_array.shape)
plt.imshow(img_array)
plt.show()
model = BaselineAutoencoder()
tensor = torch.tensor(img_array, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)
latent = model.forward(tensor)
print(latent.size())

