from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.train_config import TrainConfig
from scipy.stats import gaussian_kde
import yaml

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

MARGIN_RATIO = 0.5


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
    tsne = TSNE(random_state=42)
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
    k = gaussian_kde(
        (encoder_output_transformed[:, 0], encoder_output_transformed[:, 1])
    )
    margin = MARGIN_RATIO * encoder_output_transformed[:, 0].max()
    xi, yi = np.mgrid[
        encoder_output_transformed[:, 0].min()
        - margin : encoder_output_transformed[:, 0].max()
        + margin : n_bins * 1j,
        encoder_output_transformed[:, 1].min()
        - margin : encoder_output_transformed[:, 1].max()
        + margin : n_bins * 1j,
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap="Reds")
    return


def scatter_codebooks(
    codebook_transformed: np.array,
    ax: plt.Axes,
    codebook_partitioning: dict = None,
):
    if codebook_partitioning is None:
        ax.scatter(
            codebook_transformed[:, 0],
            codebook_transformed[:, 1],
            alpha=0.5,
            s=8,
        )
    else:
        for name, partition in codebook_partitioning.items():
            ax.scatter(
                codebook_transformed[partition[0] : partition[1], 0],
                codebook_transformed[partition[0] : partition[1], 1],
                alpha=0.5,
                s=8,
                label=name,
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


def main():
    """image_path = "auto2.jpg"
    image_path = image_path  # Replace with your image path
    image = Image.open(image_path).convert("RGB")  # Ensure 3 color channels (RGB)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Start loading...")
    test_loader, _ = load_data(
        "imagenet", n_train=1000, n_test=1000, seed=42, batch_size=64
    )
    print("Loaded.")
    image_tensor = next(iter(test_loader))[0]
    image_tensor.to(DEVICE)

    encoder_outputs = []
    codebooks = []
    for epoch in ["0", "4", "21"]:
        model = load_model(
            f"artifacts/hyperopt_X_with_replacement_43/model_{epoch}.pth",
            "eval_config.yaml",
        )
        model.to(DEVICE)
        codebooks.append(model.codebook.detach().cpu().numpy())
        print(f"Shape: {image_tensor.shape}")
        encoder_output = model.encoder(image_tensor).detach().cpu().numpy()
        encoder_outputs.append(sample(np.reshape(encoder_output, (-1, 256)), 2048))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    codebooks_transformed, encoder_outputs_transformed = transform_data(
        codebooks, encoder_outputs
    )
    for epoch in range(0, 3):
        plot_encoder_density(encoder_outputs_transformed[epoch], axs[epoch])
        scatter_codebooks(codebooks_transformed[epoch], axs[epoch])
    fig.suptitle("TEST")
    plt.savefig("test.jpg")

    plt.show()


if __name__ == "__main__":
    main()
