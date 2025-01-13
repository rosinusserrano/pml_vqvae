from pml_vqvae.train_config import TrainConfig
from scipy.stats import gaussian_kde
import yaml

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

MARGIN_SIZE = 3


def load_model(model_path, config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = TrainConfig.from_dict(yaml.safe_load(file))
    model = config.get_model()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    return model


def save_2d_tsne(
    codebook: np.array,
    filename: str = "eval_2d_sne.png",
    title: str = "t-SNE Visualization of Codebook Embeddings",
    range_dict: dict = None,
    encoder_output: np.array = None,
):
    plt.figure(figsize=(8, 6))
    tsne = TSNE(random_state=42)
    if encoder_output is not None:
        transformed_data = tsne.fit_transform(np.concat([codebook, encoder_output]))

        nbins = 16
        x = transformed_data[len(codebook) :, 0]
        y = transformed_data[len(codebook) :, 1]
        print(x.shape)
        k = gaussian_kde((x, y))
        xi, yi = np.mgrid[
            transformed_data[:, 0].min()
            - MARGIN_SIZE : transformed_data[:, 0].max()
            + MARGIN_SIZE : nbins * 1j,
            transformed_data[:, 1].min()
            - MARGIN_SIZE : transformed_data[:, 1].max().max()
            + MARGIN_SIZE : nbins * 1j,
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap="Blues")

        cbar = plt.colorbar()
        cbar.set_label("Encoder Output Density")
    else:
        transformed_data = tsne.fit_transform(codebook)

    # plot the codebooks
    if range_dict is None:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    else:
        for name, partition in range_dict.items():
            plt.scatter(
                transformed_data[partition[0] : partition[1], 0],
                transformed_data[partition[0] : partition[1], 1],
                alpha=0.5,
                s=16,
                label=name,
            )

        plt.legend(loc="lower left")
    plt.xlim(
        left=transformed_data[:, 0].min() - MARGIN_SIZE,
        right=transformed_data[:, 0].max() + MARGIN_SIZE,
    )
    plt.ylim(
        bottom=transformed_data[:, 1].min() - MARGIN_SIZE,
        top=transformed_data[:, 1].max() + MARGIN_SIZE,
    )

    plt.title(title)
    plt.savefig(filename)
    plt.show()


def create_range(codebooks_dict):
    range_dict = {}
    codebook_list = []
    curr = 0
    for name, codebook in codebooks_dict.items():
        range_dict[name] = (curr, curr + len(codebook))
        curr += len(codebook)
        codebook_list.append(codebook)
    concat_codebooks = np.concat(codebook_list)
    return concat_codebooks, range_dict


def main():
    m0_codebook = (
        load_model("model_0.pth", "eval_config.yaml").codebook.detach().numpy()[0:512]
    )
    m21_codebook = (
        load_model("model_21.pth", "eval_config.yaml").codebook.detach().numpy()[0:512]
    )
    m4_codebook = (
        load_model("model_4.pth", "eval_config.yaml").codebook.detach().numpy()[0:512]
    )

    # Replace with using the models
    encoder_output = 2 * np.random.randn(1000, 256)

    concat_codebooks, range_dict = create_range(
        {"model 0": m0_codebook, "model 4": m4_codebook, "model 21": m21_codebook}
    )
    save_2d_tsne(concat_codebooks, range_dict=range_dict, encoder_output=encoder_output)


if __name__ == "__main__":
    main()
