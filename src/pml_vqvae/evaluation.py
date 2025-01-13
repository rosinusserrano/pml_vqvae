from pml_vqvae.train_config import TrainConfig
import yaml

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

codebook_np = np.random.random((128, 256))


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
):
    tsne = TSNE(random_state=42)
    codebook_2d = tsne.fit_transform(codebook)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if range_dict is None:
        ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1])
    else:
        for key, value in range_dict.items():
            ax.scatter(
                codebook_2d[value[0] : value[1], 0],
                codebook_2d[value[0] : value[1], 1],
                alpha=0.4,
                s=16,
                label=key,
            )
        plt.legend(loc="lower left")

    plt.title(title)
    plt.grid(True)
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
    )


if __name__ == "__main__":
    main()
