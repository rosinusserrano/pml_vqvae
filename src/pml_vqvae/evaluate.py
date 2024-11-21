"Functions to evaluate models"

from typing import Callable, Literal
from functools import partial

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.utils import make_grid

from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt

from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show_image_grid, show_image_grid_v2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def plot_original_and_reconstruction(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    ncols: int = 8,
    mode: Literal["horizontal", "vertical"] = "vertical",
    outfile: str | None = None,
):
    "Produce a plot that shows images side by side"
    orig_grid = make_grid(original_images, nrow=ncols, padding=1)
    recon_grid = make_grid(reconstructed_images, nrow=ncols, padding=1)

    grid = make_grid(
        torch.stack((orig_grid, recon_grid)),
        padding=2,
        nrow=(2 if mode == "horizontal" else 1),
    )

    if outfile is not None:
        plt.imsave(outfile, grid.permute(1, 2, 0).detach().cpu().numpy())
    else:
        plt.imshow(grid.permute(1, 2, 0))


def batch_structural_similarity(
    original_batch: torch.Tensor, reconstruction_batch: torch.Tensor
):
    """Compute SSIM (scikit learn impl.) on a PyTorch batch of images."""
    ssim_batch = []
    batch_size = original_batch.shape[0]
    for idx in range(batch_size):
        original, reconstruction = original_batch[idx], reconstruction_batch[idx]
        ssim = structural_similarity(
            original.detach().cpu().numpy(),
            reconstruction.detach().cpu().numpy(),
            channel_axis=0,
            data_range=1.0,
        )
        ssim_batch.append(ssim)

    return sum(ssim_batch) / len(ssim_batch)


def evaluate_on_class(
    model: nn.Module,
    dataset: str,
    n_samples: int,
    batch_size: int,
    transform: Callable,
    class_idx: int,
    evaluation_metric: Callable,
    break_after_first_batch: bool = False,
):
    _, dataloader = load_data(
        dataset,
        transformation=transform,
        batch_size=batch_size,
        n_test=n_samples,
        n_train=1000 if dataset == "imagenet" else 10,
        class_idx=[class_idx],
    )

    metric_values = []

    with eval_mode(model):
        for batch, labels in dataloader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(batch)

            metric_value = evaluation_metric(labels, preds)
            if metric_value is not None:
                metric_values.append(metric_value)

            if break_after_first_batch:
                break

    avg_metric_value = None
    if len(metric_values) != 0:
        avg_metric_value = sum(metric_values) / len(metric_values)

    return avg_metric_value


if __name__ == "__main__":

    print("Loading model")

    model_file_path = "artifacts/[FINAL] vae imagenet 10 epochs_output/model_10.pth"

    model = BaselineVariationalAutoencoder()
    model.load_state_dict(
        torch.load(model_file_path, weights_only=True, map_location=torch.device("cpu"))
    )
    model.to(DEVICE)

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ]
    )

    print("Evaluating")

    avg_ssim_on_class_0 = evaluate_on_class(
        model, "imagenet", 1000, 32, transforms, 0, plot_original_and_reconstruction
    )

    print(f"On class 0 got avg SSIM of {avg_ssim_on_class_0:.4f}")
