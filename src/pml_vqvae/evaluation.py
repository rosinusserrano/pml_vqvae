from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.train_config import TrainConfig
from scipy.stats import gaussian_kde
import yaml

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import math

from PIL import Image
from torchvision import transforms

MARGIN_RATIO = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = TrainConfig.from_dict(yaml.safe_load(file))
    model = config.get_model()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    return model


def transform_data(codebooks, encoder_outputs=None):
    codebooks_combined = np.concat(codebooks)
    if encoder_outputs is not None:
        encoder_outputs_combined = np.concat(encoder_outputs)
        all_data = np.concat([codebooks_combined, encoder_outputs_combined])
    else:
        all_data = codebooks_combined
    tsne = TSNE(random_state=42, n_jobs=-1)
    transformed_data = tsne.fit_transform(all_data)
    transformed_codebooks = []
    start = 0
    for codebook in codebooks:
        end = start + len(codebook)
        transformed_codebooks.append(transformed_data[start:end])
        start = end
    if encoder_outputs is not None:
        transformed_encoder_outputs = []
        for encoder_output in encoder_outputs:
            end = start + len(encoder_output)
            transformed_encoder_outputs.append(transformed_data[start:end])
            start = end
    else:
        transformed_encoder_outputs = None
    return transformed_codebooks, transformed_encoder_outputs


def plot_encoder_density(encoder_output_transformed, ax):
    n_bins = 128
    lower_limit, upper_limit = (
        np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
        np.max([ax.get_xlim()[1], ax.get_ylim()[1]]),
    )

    ticks = {
        "lower": math.ceil(lower_limit / 10) * 10,
        "upper": math.ceil(upper_limit / 10) * 10,
    }
    if ticks["lower"] * -1 < ticks["upper"]:
        ticks["upper"] = ticks["lower"] * -1
    else:
        ticks["lower"] = ticks["upper"] * -1
    ax.set_xticks(
        [ticks["lower"], 0.5 * ticks["lower"], 0, 0.5 * ticks["upper"], ticks["upper"]]
    )
    ax.set_yticks(
        [ticks["lower"], 0.5 * ticks["lower"], 0, 0.5 * ticks["upper"], ticks["upper"]]
    )

    margin = (upper_limit - lower_limit) * MARGIN_RATIO
    lower_limit, upper_limit = lower_limit - margin, upper_limit + margin

    k = gaussian_kde(
        (encoder_output_transformed[:, 0], encoder_output_transformed[:, 1])
    )
    xi, yi = np.mgrid[
        lower_limit : upper_limit : n_bins * 1j,
        lower_limit : upper_limit : n_bins * 1j,
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.pcolormesh(
        xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap="Reds", zorder=0
    )
    return


def scatter_codebooks(
    codebook_transformed: np.array,
    ax: plt.Axes,
    codebook_partitioning: dict = None,
):
    filled_marker_style = dict(
        marker="o", color="cyan", edgecolor="black", alpha=0.45, s=12
    )
    if codebook_partitioning is None:
        ax.scatter(
            codebook_transformed[:, 0],
            codebook_transformed[:, 1],
            **filled_marker_style,
        )
    else:
        for name, partition in codebook_partitioning.items():
            ax.scatter(
                codebook_transformed[partition[0] : partition[1], 0],
                codebook_transformed[partition[0] : partition[1], 1],
                alpha=0.5,
                label=name,
                **filled_marker_style,
            )
        ax.legend(loc="lower left")


def create_codebook_partitioning(codebooks_dict):
    range_dict = {}
    codebook_list = []
    curr = 0
    for name, codebook in codebooks_dict.items():
        range_dict[name] = (curr, curr + len(codebook))
        curr += len(codebook)
        codebook_list.append(codebook)
    concat_codebooks = np.concat(codebook_list)
    return concat_codebooks, range_dict


def sample(array, num_samples):
    random_indices = np.random.choice(array.shape[0], size=num_samples, replace=False)
    return array[random_indices]


def plot_multiple_classes(model_name, output_img_name, model_dir=".", class_idxs=None):
    print("Loading dataset.")
    data_loaders = []
    for class_idx in class_idxs:
        _, test_loader = load_data(
            "imagenet",
            n_train=1000,
            n_test=1000,
            seed=42,
            batch_size=512,
            class_idx=[class_idx] if class_idx is not None else None,
        )
        data_loaders.append(test_loader)

    print("Dataset loaded.")

    image_tensors = []
    for no_class_idx, _ in enumerate(class_idxs):
        image_tensor = next(iter(data_loaders[no_class_idx]))[0]
        image_tensor.to(DEVICE)
        image_tensors.append(image_tensor)

    encoder_outputs = []
    codebooks = []
    model = load_model(
        f"{model_dir}/{model_name}.pth",
        "eval_config.yaml",
    )
    model.to(DEVICE)
    with torch.no_grad():
        codebooks.append(model.codebook.detach().cpu().numpy())
        for image_tensor in image_tensors:
            encoder_output = (
                model.encoder(image_tensor).detach().cpu().permute(3, 2, 0, 1).numpy()
            )

            encoder_outputs.append(sample(np.reshape(encoder_output, (-1, 256)), 10000))
    codebooks_transformed, encoder_outputs_transformed = transform_data(
        codebooks, encoder_outputs
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    for class_idx_no, class_idx in enumerate(class_idxs):
        ax = axs[class_idx_no]
        ax.set_aspect(1)
        ax.tick_params(axis="both", labelsize=16, length=10, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        scatter_codebooks(codebooks_transformed[0], ax)
        plot_encoder_density(encoder_outputs_transformed[class_idx_no], ax)
    plt.tight_layout()
    plt.savefig(f"{output_img_name}.png")

    plt.show()


def plot_from_dataset(model_names, output_img_name, model_dir="."):
    print("Loading dataset.")
    _, test_loader = load_data(
        "imagenet", n_train=1000, n_test=1000, seed=42, batch_size=512
    )
    print("Dataset loaded.")
    image_tensor = next(iter(test_loader))[0]
    image_tensor.to(DEVICE)
    encoder_outputs = []
    codebooks = []
    for model in model_names:
        model = load_model(
            f"{model_dir}/{model}.pth",
            "eval_config.yaml",
        )
        model.to(DEVICE)
        with torch.no_grad():
            codebooks.append(model.codebook.detach().cpu().numpy())
            encoder_output = (
                model.encoder(image_tensor).detach().cpu().permute(3, 2, 0, 1).numpy()
            )
            encoder_outputs.append(sample(np.reshape(encoder_output, (-1, 256)), 10000))
    codebooks_transformed, encoder_outputs_transformed = transform_data(
        codebooks, encoder_outputs
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for model_no, _ in enumerate(model_names):
        ax = axs[model_no]
        ax.set_aspect(1)
        ax.tick_params(axis="both", labelsize=16, length=10, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        scatter_codebooks(codebooks_transformed[model_no], ax)
        plot_encoder_density(encoder_outputs_transformed[model_no], ax)
    plt.tight_layout()
    plt.savefig(f"{output_img_name}.png")

    plt.show()


def plot_from_image(model_names, output_img_name, img_path="auto2.jpg", model_dir="."):
    image = Image.open(img_path).convert("RGB")  # Ensure 3 color channels (RGB)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    image_tensor.to(DEVICE)
    encoder_outputs = []
    codebooks = []
    for model in model_names:
        model = load_model(
            f"{model_dir}/{model}.pth",
            "eval_config.yaml",
        )
        model.to(DEVICE)
        with torch.no_grad():
            codebooks.append(model.codebook.detach().cpu().numpy())
            encoder_output = (
                model.encoder(image_tensor).detach().cpu().permute(3, 2, 0, 1).numpy()
            )
            encoder_outputs.append(sample(np.reshape(encoder_output, (-1, 256)), 256))
    codebooks_transformed, encoder_outputs_transformed = transform_data(
        codebooks, encoder_outputs
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for model_no, _ in enumerate(model_names):
        ax = axs[model_no]
        ax.set_aspect(1)
        ax.tick_params(axis="both", labelsize=16, length=10, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        scatter_codebooks(codebooks_transformed[model_no], ax)
        plot_encoder_density(encoder_outputs_transformed[model_no], ax)
    plt.tight_layout()
    plt.savefig(f"{output_img_name}.png")

    plt.show()


def main():
    plot_multiple_classes(
        "model_2",
        "worst_all_best",
        "artifacts/FINAL REPLACEMENT VQVAE",
        class_idxs=[530, None, 550],
    )


if __name__ == "__main__":
    main()
