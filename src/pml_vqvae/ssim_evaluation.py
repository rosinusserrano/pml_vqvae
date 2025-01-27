"Functions to evaluate models"

from typing import Callable, Literal
from functools import partial
import yaml

import numpy as np

from PIL import Image
import os

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.utils import make_grid

from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt

from tqdm import trange
from tqdm.auto import tqdm

from pml_vqvae.models.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.models.baseline.vae import BaselineVAE
from pml_vqvae.models.vqvae import (
    VQVAE,
    VQVAECodeEnforced,
    VQVAECodeEnforcedConfig,
    VQVAEConfig,
)
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str, model_config_dict: dict | None):
    if model_name == "vqvae":
        if model_config_dict is None:
            raise ValueError("VQ-VAE needs model config!")
        config = VQVAEConfig(**model_config_dict)
        return VQVAE(config)

    if model_name == "vqvae-ce":
        if model_config_dict is None:
            raise ValueError("VQ-VAE needs model config!")
        config = VQVAECodeEnforcedConfig(**model_config_dict)
        return VQVAECodeEnforced(config)

    if model_name == "pixelcnn":
        if model_config_dict is None:
            raise ValueError("PixelCNN needs model config!")
        config = PixelCNNConfig(**model_config_dict)
        return PixelCNN(config)

    raise ValueError(f"Model {model_name} is not available.")


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
        show(original_images, outfile if outfile is None else f"original_{outfile}")
        show(
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


def plot_image_comparison_of_class(
    model: nn.Module,
    class_idx: int,
    grid_shape_per_class: tuple[int, int] = (4, 2),
    mode: Literal["horizontal", "vertical"] = "vertical",
):
    n_samples = grid_shape_per_class[0] * grid_shape_per_class[1]

    model.to(DEVICE)

    _, dataloader = load_data(
        "imagenet",
        batch_size=n_samples,
        n_test=n_samples,
        n_train=0,
        class_idx=[class_idx],
    )

    images = next(iter(dataloader))[0].to(DEVICE)

    reconstructions = model(images)

    plot_original_and_reconstruction(
        images,
        reconstructions,
        ncols=grid_shape_per_class[1],
        mode=mode,
        outfile=f"image_comparison_class_{class_idx}.png",
    )


def batch_structural_similarity(
    original_batch: torch.Tensor,
    reconstruction_batch: torch.Tensor,
    reduce: bool = True,
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

    if reduce:
        return sum(ssim_batch) / len(ssim_batch)

    return ssim_batch


def ssim_for_all_classes(
    model: nn.Module,
    dataset: str,
    n_samples: int | None = None,
):
    """Compute average SSIM score for each class.

    n_samples will be the number of samples drawn from the test set. Thus the
    number of samples used per class will be n_samples / n_classes. If None,
    all data is used. Imagenet test set is 100k samples.
    """
    _, dataloader = load_data(
        dataset,
        batch_size=32,
        n_test=n_samples,
        n_train=0,
    )

    metric_values = {idx: [] for idx in range(1000)}

    with eval_mode(model):
        for i, (batch, labels) in enumerate(tqdm(dataloader)):
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(batch)
            if isinstance(preds, tuple):
                preds = preds[0]

            batch_ssim = batch_structural_similarity(batch, preds, reduce=False)
            for i in range(len(batch_ssim)):
                metric_values[labels[i].long().item()].append(batch_ssim[i])

    averages = {}
    for idx in metric_values.keys():
        averages[idx] = sum(metric_values[idx]) / len(metric_values[idx])

    return averages


def get_tsne_data_for_class(
    vqvae: VQVAE,
    class_idx: int,
    batch_size: int = 1,
    hyperclass: bool = False,
):
    """This function returns the encoder output for one batch of a given class.

    The output is reshaped such that the spatial dimensions are collapsed,
    meaning that for imagent, the encoder output is reshaped from batch_size x
    32 x 32 x embedding_dimension to batch_size*1024 x embedding_dimension.
    Additionally the output is converted to a numpy array.
    """
    _, dataloader = load_data(
        "imagenet",
        n_train=0,
        n_test=None,
        batch_size=batch_size,
        class_idx=[class_idx],
        hyperclass=hyperclass,
    )

    batch = next(iter(dataloader))[0].to(DEVICE)

    encoder_out = vqvae.encoder(batch)

    encoder_out = encoder_out.detach().permute(0, 2, 3, 1)
    encoder_out = torch.reshape(encoder_out, (-1, encoder_out.shape[-1]))
    return encoder_out.detach().cpu().numpy()


def make_ssim_boxplots(
    models: list[PML_model],
    model_names: list[str],
    dataset: str = "imagenet",
    save_ssim_dict: bool = True,
):

    ssims_dict = {}
    for model in models:
        ssim = ssim_for_all_classes(model, dataset, n_samples=None)

        ssims_dict[model_name] = ssim

        print(
            f"{model.name()} got min ssim",
            min(ssim.values()),
            "on class",
            min(ssim.items(), key=lambda x: x[1])[0],
        )
        print(
            f"{model.name()} got max ssim",
            max(ssim.values()),
            "on class",
            max(ssim.items(), key=lambda x: x[1])[0],
        )

    if save_ssim_dict:
        with open("avg_ssim_per_class_per_model.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(ssims_dict, f)

    plt.boxplot(
        list(map(lambda k: list(ssims_dict[k].values())), ssims_dict.keys()),
        tick_labels=model_names,
    )
    plt.savefig("boxplot_ssim.png")

    return ssims_dict


def jpeg_compression_check():
    for q in [90, 92, 93, 94, 95, 96, 97, 98, 99, 100]:
        bits_per_pixels = []

        print(f"JPEG quality: {q}")

        for fname in os.listdir("sample_images/pngs"):
            img: Image.Image = Image.open(f"sample_images/pngs/{fname}")
            img.save("jpeg.jpg", "JPEG", quality=q)
            filesize = os.path.getsize("jpeg.jpg") * 8
            bits_per_pixels.append(filesize / (img.width * img.height))

        avg = sum(bits_per_pixels) / len(bits_per_pixels)

        print(f"Avg bits per pixel {avg}")


def png_jjpeg100_ssim():
    ssims = []
    for fname in os.listdir("sample_images/pngs"):
        img: Image.Image = Image.open(f"sample_images/pngs/{fname}")
        img.save("jpeg.jpg", "JPEG", quality=97)
        jpg = Image.open("jpeg.jpg")

        ssims.append(
            structural_similarity(
                np.asarray(img),
                np.asarray(jpg),
                channel_axis=2 if len(np.asarray(img).shape) == 3 else None,
                data_range=255.0,
            )
        )

    print(sum(ssims) / len(ssims))


if __name__ == "__main__":
    MODEL_PATHS = [
        "artifacts/final_replacement_vqvae",
        "artifacts/final_codeenforced_vqvae",
    ]

    models = []
    model_names = []

    for p in MODEL_PATHS:
        config_file = f"{p}/config.yaml"
        model_file = f"{p}/model.pth"

        print("Reading config file")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        model_config_dict = config_dict["model_config"]["value"]
        model_name = config_dict["model_name"]["value"]

        model_names.append(model_name)

        print(f"Loading model {model_name}")
        model: VQVAE = get_model(model_name, model_config_dict)
        model.load_state_dict(
            torch.load(model_file, weights_only=True, map_location="cpu")
        )
        model.to(DEVICE)

        models.append(model)

    print("Get SSIM per class")
    avg_ssim_per_model = make_ssim_boxplots(models, model_names)

    for model_name in avg_ssim_per_model.keys():
        print(f"Results for {model_name}")
        avg_ssim = avg_ssim_per_model[model_name]
        ordered_ssim_values = sorted(avg_ssim.items(), key=lambda x: x[1])

        n_classes = len(ordered_ssim_values)

        best_class = ordered_ssim_values[-1]
        median_class = ordered_ssim_values[n_classes // 2]
        worst_class = ordered_ssim_values[0]

        print("best", best_class, "median", median_class, "worst", worst_class)

        encoder_out = get_tsne_data_for_class(model, best_class[0], batch_size=32)

        print(encoder_out.shape)
