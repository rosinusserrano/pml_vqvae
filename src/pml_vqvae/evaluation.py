from pml_vqvae.train_config import TrainConfig
import yaml

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

codebook_np = np.random.random((128, 256))


# Example list of 2D data points


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
    filename: str = "eval_2d_sne.svg",
    title: str = "t-SNE Visualization of Codebook Embeddings",
    range_dict: dict = None,
    encoder_output: np.array = None,
):
    plt.figure(figsize=(8, 6))
    tsne = TSNE(random_state=42)
    if encoder_output is not None:
        transformed_data = tsne.fit_transform(np.concat([codebook, encoder_output]))
        hist = plt.hist2d(
            transformed_data[len(codebook) :, 0],
            transformed_data[len(codebook) :, 1],
            bins=64,
            cmap="Blues",
            alpha=0.8,
        )  # only put the transformed encoder_output into the heatmap, so [len(codebook):, 0]
        plt.colorbar(hist[3], label="Density")
        plt.title("Density Plot using hist2d")
        plt.xlabel("X")
        plt.ylabel("Y")
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
    plt.axis("auto")

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
        load_model("model_0.pth", "eval_config.yaml").codebook.detach().numpy()
    )
    m21_codebook = (
        load_model("model_21.pth", "eval_config.yaml").codebook.detach().numpy()
    )
    m4_codebook = (
        load_model("model_4.pth", "eval_config.yaml").codebook.detach().numpy()
    )

    concat_codebooks, range_dict = create_range(
        {"model 0": m0_codebook, "model 4": m4_codebook, "model 21": m21_codebook}
    )
    save_2d_tsne(
        concat_codebooks,
        range_dict=range_dict,
        encoder_output=np.random.randn(10000, 256),
    )


if __name__ == "__main__":
    main()
