"Functions to evaluate models"

from typing import Callable

import torch
from torch import nn
from torchvision.transforms import v2

from skimage.metrics import structural_similarity

from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
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


def image_human_evaluation(batch, labels, preds):
    "Produce a plot that shows images side by side"


def batch_structural_similarity(
    original_batch: torch.Tensor, reconstruction_batch: torch.Tensor
):
    """Compute SSIM (scikit learn impl.) on a PyTorch batch of images."""
    ssim_batch = []
    batch_size = original_batch.shape[0]
    for idx in range(batch_size):
        original, reconstruction = original_batch[idx], reconstruction_batch[idx]
        ssim = structural_similarity(
            original.permute(1, 2, 0).detach().cpu().numpy(),
            reconstruction.permute(1, 2, 0).detach().cpu().numpy(),
        )
        ssim_batch.append(ssim)


def evaluate_on_class(
    model: nn.Module,
    dataset: str,
    n_samples: int,
    transform: Callable,
    class_idx: int,
    evaluation_metric: Callable,
):
    _, dataloader = load_data(
        dataset,
        transformation=transform,
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

            metric_value = evaluation_metric(batch, labels, preds)
            if metric_value is not None:
                metric_values.append(metric_value)

    avg_metric_value = None
    if len(metric_values) != 0:
        avg_metric_value = sum(metric_values) / len(metric_values)

    return avg_metric_value


if __name__ == "__main__":

    model_file_path = "artifacts/model_3.pth"

    model = BaselineAutoencoder()
    model.load_state_dict(
        torch.load(model_file_path, weights_only=True, map_location=torch.device("cpu"))
    )

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ]
    )

    avg_ssim_on_class_0 = evaluate_on_class(
        model, "imagenet", 10000, transforms, 0, batch_structural_similarity
    )
