import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
from pml_vqvae.dataset.dataloader import load_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# get vqvae
print("Loading VQVAE")
vqvaeconfig = VQVAEConfig(
    codebook_size=1024,
    commitment_weight=2,
    hidden_dimension=64,
    embedding_dimension=64,
)
vqvae = VQVAE(vqvaeconfig).to(DEVICE)
vqvae.load_state_dict(
    torch.load("artifacts/vqvae_konni_easy_params/model.pth", weights_only=True)
)

# get pixelcnn
print("Loading PixelCNN")
pixelcnnconfig = PixelCNNConfig(
    num_codes=1024,
    hidden_chan=256,
    num_classes=30,
    input_shape=(32, 32),
)
pixelcnn = PixelCNN(pixelcnnconfig).to(DEVICE)
pixelcnn.load_state_dict(
    torch.load(
        "artifacts/pixelcnn on 30 classes test_4/model_50.pth", weights_only=True
    )
)


# function for constructing 2 4x2 grids of original/reconstruction
def original_next_to_reconstruction(class_idx):
    trainloader, testloader = load_data(
        "imagenet",
        None,
        None,
        num_workers=2,
        batch_size=8,
        class_idx=[class_idx],
    )

    original, _ = next(iter(testloader))
    reconstructed, _, _, _ = vqvae(original.to(DEVICE))

    original_grid = make_grid(
        original,
        nrow=2,
        padding=1,
        pad_value=1,
        normalize=True,
    )

    reconstructed_grid = make_grid(
        reconstructed.detach().cpu(),
        nrow=2,
        padding=1,
        pad_value=1,
        normalize=True,
    )

    return make_grid(
        torch.stack((original_grid, reconstructed_grid)),
        nrow=2,
        padding=2,
        pad_value=1,
    )


# function for making 4x4 grid of generations
def generate_images(class_idx):
    indices = pixelcnn.sample((torch.ones(16) * class_idx).long().to(DEVICE))
    generated_images = vqvae.decode(indices.long())
    return make_grid(
        generated_images.detach().cpu(),
        padding=1,
        nrow=4,
        pad_value=1,
        normalize=True,
    )


# function for plotting 3 grids side by side
def plot_3_grids(
    gridA: torch.Tensor,
    gridB: torch.Tensor,
    gridC: torch.Tensor,
    outfile: str,
):

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))

    ax[0].imshow(gridA.permute(1, 2, 0).detach().cpu().numpy())
    ax[0].axis("off")
    ax[1].imshow(gridB.permute(1, 2, 0).detach().cpu().numpy())
    ax[1].axis("off")
    ax[2].imshow(gridC.permute(1, 2, 0).detach().cpu().numpy())
    ax[2].axis("off")

    fig.savefig(outfile, bbox_inches="tight")


# MAIN
if __name__ == "__main__":
    mse_classes = [29, 24, 3]
    ssim_classes = [29, 24, 6]
    fid_classes = [4, 10, 9]

    print("Creating mse images")
    mse_grids = [original_next_to_reconstruction(idx) for idx in mse_classes]
    plot_3_grids(*mse_grids, "mse_grids.svg")

    print("Creating ssim images")
    ssim_grids = [original_next_to_reconstruction(idx) for idx in ssim_classes]
    plot_3_grids(*ssim_grids, "ssim_grids.svg")

    print("Creating fid images")
    fid_grids = [generate_images(idx) for idx in fid_classes]
    plot_3_grids(*fid_grids, "fid_grids.svg")
