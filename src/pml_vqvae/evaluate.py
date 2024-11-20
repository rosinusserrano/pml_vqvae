"Functions to evaluate models"

import torch
from torch import nn

from torchvision.transforms import v2

from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show_image_grid, show_image_grid_v2


class eval_mode:
    def __init__(self, model: nn.Module):
        self.grad_previously_enabled = False
        self.previous_training_mode = model.training
        self.model = model

    def __enter__(self):
        self.grad_previously_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self.model.train(False)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.grad_previously_enabled)
        self.model.train(self.previous_training_mode)


def generate_images(
    model: nn.Module,
    latent_size: tuple[int, ...],
    num_images: int,
    outfile: str | None = None,
):
    """"""
    with eval_mode(model):
        z = torch.randn((num_images, *latent_size))
        x = model(z)
        show_image_grid_v2(x, outfile)


def reconstruct_images(
    model: nn.Module, original_images: torch.Tensor, outfile: str | None = None
):
    with eval_mode(model):
        reconstruction = model(original_images)
        show_image_grid_v2(
            original_images, outfile if outfile is None else f"original_{outfile}"
        )
        show_image_grid_v2(
            reconstruction, outfile if outfile is None else f"reconstruction_{outfile}"
        )


if __name__ == "__main__":
    model_file_path = "artifacts/model_3.pth"
    model = BaselineAutoencoder()
    model.load_state_dict(
        torch.load(model_file_path, weights_only=True, map_location=torch.device("cpu"))
    )

    # trainl, testl = load_data(
    #     dataset="imagenet",
    #     n_train=1000,
    #     n_test=1000,
    #     batch_size=64,
    #     seed=42,
    #     transformation=v2.Compose(
    #         [
    #             v2.CenterCrop(128),
    #             v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
    #         ]
    #     ),
    # )

    batch = torch.randn((32, 3, 128, 128))
    reconstruct_images(model, batch)

    generate_images(model.decoder_stack, (2, 32, 32), 32)
