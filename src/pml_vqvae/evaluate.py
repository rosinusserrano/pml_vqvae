"Functions to evaluate models"

from typing import Callable, Literal
from functools import partial

from PIL.Image import Image
import os

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.utils import make_grid

from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt

from tqdm import trange

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
        plt.imsave(outfile, grid.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy())
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


def get_jpeg_bits_per_pixel(images_batch: torch.Tensor, ):
    """sas"""
    v2.functional.


def evaluate_on_class(
    model: nn.Module,
    dataset: str,
    class_idx: int,
    evaluation_metric: Callable,
    n_samples: int | None = None,
    batch_size: int = 32,
    transform: Callable | None = None,
    break_after_first_batch: bool = False,
): 
    _, dataloader = load_data(
        dataset,
        transformation=transform,
        batch_size=batch_size,
        n_test=n_samples,
        n_train=0,
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


def make_ssim_boxplots(models):

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True, scale=(1.0, 1.0)),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ]
    )

    ssims_arr = []
    for model in models:
        ssims = []

        for class_idx in trange(1000):
            ssim = evaluate_on_class(
                model,
                "imagenet",
                class_idx,
                batch_structural_similarity,
                n_samples=10000000,
                batch_size=128,
                transform=transforms,
            )

            ssims.append(ssim)

        ssims_arr.append(ssims)

        print(f"{model.name()} got min ssim", min(ssims), "on class", torch.argmin(torch.tensor(ssims)))
        print(f"{model.name()} got max ssim", max(ssims), "on class", torch.argmax(torch.tensor(ssims)))
    
    plt.boxplot(ssims_arr, tick_labels=["Autoencoder", "VAE"])
    plt.savefig("boxplot_ae_and_vae.png")

    plt.clf()

    plt.boxplot([ssims_arr[0]])
    plt.tick_params(bottom=False, labelbottom=False)
    plt.savefig("boxplot_only_ae.png")


def image_comparison_plots(models, class_idx_dict):
    example_class_idx_dict = {
        "MIN_AE_IDX": (607, ae),
        "MAX_AE_IDX": (896, ae),
        "MIN_VAE_IDX": (509, vae),
        "MAX_VAE_IDX": (405, vae)
    }

    for idx_name, (idx, model) in class_idx_dict.items():
        print(idx_name, idx)
        evaluate_on_class(model,
            "imagenet", idx,
            partial(plot_original_and_reconstruction, outfile=f"{idx_name}_{idx}.png"),
            batch_size=32,
            transform=transforms,
            break_after_first_batch=True
        )


if __name__ == "__main__":

    # print("Loading models")

    # ae_fp = "artifacts/[FINAL] autoencoder imagenet 10 epochs_output/model_10.pth"
    # ae = BaselineAutoencoder()
    # ae.load_state_dict(
    #     torch.load(ae_fp, weights_only=True, map_location=torch.device("cpu"))
    # )
    # ae.to(DEVICE)

    # vae_fp = "artifacts/[FINAL] vae imagenet 10 epochs_output/model_10.pth"
    # vae = BaselineVariationalAutoencoder()
    # vae.load_state_dict(
    #     torch.load(vae_fp, weights_only=True, map_location=torch.device("cpu"))
    # )
    # vae.to(DEVICE)

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True, scale=(1.0, 1.0)),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ]
    )

    _, dataloader = load_data(
        "imagenet",
        transformation=transforms,
        batch_size=64,
        n_test=1000,
        n_train=0,
    )

    pil = v2.ToPILImage()

    bits_per_pixels = []

    for batch, labels in dataloader:

        images: list[Image] = pil(batch)

        for q in [40, 50, 60, 70]:

            for c in images:

                c.save("jpeg.jpg", "JPEG", quality=q)
                filesize = os.path.getsize("jpeg.jpg") * 8
                bits_per_pixels.append(filesize / (128*128*3))

            avg = sum(bits_per_pixels) / len(bits_per_pixels)

            print(f"Avg bits per pixel {avg}")



