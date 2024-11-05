import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class BaselineAutoencoder(torch.nn.Module):
    def __init__(self):
        hidden_units = 128
        super().__init__()
        self.encoder_downsampling = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=hidden_units, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        self.decoder_upsampling = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=hidden_units, out_channels=3, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        latent = self.encoder_downsampling(x)
        reconstruction = self.decoder_upsampling(latent)
        reconstruction = torch.clamp(reconstruction, 0., 1.)
        return reconstruction

def simple_downsample(image, scale):
    new_img = np.zeros((128, 128, 3), dtype=np.float32)
    for i in range(0, 128):
        for j in range(0, 128):
            new_img[i][j] = image[i * scale][j * scale] / 255.
    return new_img

def main():
    plt.figure(figsize=(3,3))

    img_array = np.array(Image.open('data/os361x7rfy151.webp'))
    img_array = simple_downsample(img_array, 8)
    plt.imshow(img_array)
    plt.show()

    img_tensor = torch.tensor(img_array, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)
    model = BaselineAutoencoder()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for i in range(0, 5000):
        optimizer.zero_grad()
        decoded = model.forward(img_tensor)
        loss = loss_fn(img_tensor, decoded)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:

            decoded_img = decoded.detach().squeeze().permute(1, 2, 0).numpy()

            plt.figure(figsize=(3, 3))
            plt.title(f'after {i}')
            plt.imshow(decoded_img)
            plt.show()



if __name__=='__main__':
    main()