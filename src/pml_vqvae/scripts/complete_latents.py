import torch
import numpy as np

from typing import Literal

from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
from pml_vqvae.visuals import show
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.scripts.inspect_usage_of_codes import get_usage_of_codes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def omit_checkerboard(indices: torch.Tensor):
    batch_size, height, width = indices.shape
    for h in range(0, height):
        for w in range(0, width, 2):
            w_eff = w + 1 if h % 2 == 0 else w
            indices[:, h, w_eff] = -1
    return indices


def omit_half(indices: torch.Tensor, side_to_omit: Literal["t", "b", "l", "r"]):
    bs, h, w = indices.shape
    if side_to_omit == "t":
        indices[:, : h // 2, :] = -1
    if side_to_omit == "b":
        indices[:, h // 2 :, :] = -1
    if side_to_omit == "l":
        indices[:, :, : w // 2] = -1
    if side_to_omit == "r":
        indices[:, :, w // 2 :] = -1
    return indices


if __name__ == "__main__":
    print("On device", DEVICE)

    print("Loading VQVAE")
    vqvaeconfig = VQVAEConfig(
        codebook_size=512,
        commitment_weight=2,
        hidden_dimension=256,
        embedding_dimension=256,
        codebook_initialization_radius=0.5,
    )
    vqvae = VQVAE(vqvaeconfig).to(DEVICE)
    vqvae.load_state_dict(
        torch.load("artifacts/konni_replacement_vqvae/model.pth", weights_only=True)
    )

    print("Loading PixelCNN")
    pixelcnnconfig = PixelCNNConfig(
        num_codes=512,
        conditional=True,
        hidden_chan=128,
        num_classes=7,
        conditional_embedding_dim=16,
        dilations=[1, 2, 1, 4, 1, 2, 1, 2, 1],
        input_shape=(32, 32),
        use_code_embeddings=True,
        vqvae_path="artifacts/konni_replacement_vqvae",
    )
    pixelcnn = PixelCNN(pixelcnnconfig).to(DEVICE)
    pixelcnn.load_state_dict(
        torch.load(
            "artifacts/pixelcnn hyperclass code embeds_3/model_20.pth",
            weights_only=True,
        )
    )

    print("Getting samples to complete")
    testloader = load_data("imagenet", batch_size=64, hyperclass=True)[1]
    batch, labels = next(iter(testloader))
    encoded = vqvae.encode(batch.to(DEVICE))
    incomplete = omit_half(encoded, side_to_omit="b")

    print("Completing latent indices with PixelCNN")
    indices = pixelcnn.complete(incomplete, class_idx_list=labels.to(DEVICE))

    print("Generating images with decoder of VQVAE")
    generated_images = vqvae.decode(indices.long()).detach().cpu()
    show(batch, outfile="pixelcnn_input_half.png")
    show(generated_images, outfile="pixelcnn_completions_half.png")
