import os

import torch
from torch import nn
from tqdm.auto import trange
import matplotlib.pyplot as plt
import imageio
import yaml

from pml_vqvae.models.vqvae import (
    VectorQuantization,
    VQVAE,
    VQVAECodeEnforcedConfig,
    VQVAECodeEnforced,
    VQVAEConfig,
)
from pml_vqvae.visuals import show
from pml_vqvae.dataset.dataloader import load_data

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

    raise ValueError(f"Model {model_name} is not available.")


def project_onto_line(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Project c onto line between a and b."""
    ab = b - a
    ac = c - a
    t = torch.sum(ac * ab, dim=-1) / torch.sum(ab * ab, dim=-1)
    return a + t * ab


def simple_interpolate(
    a: torch.Tensor,
    b: torch.Tensor,
    steps: int,
    codebook: torch.Tensor,
):
    "Sample points on line from a to b, quantize each point."
    alpha = torch.linspace(0, 1, steps=steps)
    traj = [a]

    ab = b - a

    for i in trange(steps):
        c = a + alpha[i] * ab
        c_quant, _ = VectorQuantization.apply(c, codebook)
        traj.append(c_quant)

    return traj


# def difficult_interpolate(a: torch.Tensor, b: torch.Tensor, codebook: torch.Tensor):
#     """Discrete interpolation"""
#     bs, c, h, w = a.shape
#     assert b.shape == a.shape

#     a = a.permute(0, 2, 3, 1).reshape(-1, c)
#     b = b.permute(0, 2, 3, 1).reshape(-1, c)

#     d = torch.zeros(a.shape[:-1])

#     trajectory = [(a.copy(), d.copy())]
#     current_codes = a.copy()

#     while not torch.all(d == 1.0):
#         distance_matrix = torch.sum(
#             (current_codes[:, None, :] - codebook[None, :, :]) ** 2, dim=-1
#         )  # enc_out x cod_size

#         # HABS AUFGEGEBEN


def animate_interpolation(
    traj: list[torch.Tensor],
    decoder: nn.Module,
    outdir: str,
):
    """"""
    if not os.path.exists(f"{outdir}/frames"):
        os.makedirs(f"{outdir}/frames")

    for i, latent in enumerate(traj):
        curr_frame = decoder(latent)
        show(curr_frame, outfile=f"{outdir}/frames/{i:>08}.png")

    images = []
    for filename in sorted(os.listdir(f"{outdir}/frames")):
        images.append(imageio.imread(f"{outdir}/frames/{filename}"))
    imageio.mimsave(f"{outdir}/animation.gif", images)


if __name__ == "__main__":
    MODEL_PATH = "artifacts/final_replacement_vqvae"
    NUM_IMAGES = 64
    NUM_STEPS = 10

    config_file = f"{MODEL_PATH}/config.yaml"
    model_file = f"{MODEL_PATH}/model.pth"

    print("Reading config file...")
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    model_config_dict = config_dict["model_config"]["value"]
    model_name = config_dict["model_name"]["value"]

    print(f"Loading model {model_name}...")
    model: VQVAE = get_model(model_name, model_config_dict)
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))
    model.to(DEVICE)

    print("Loading data...")
    _, testloader = load_data(
        "imagenet",
        n_train=0,
        n_test=None,
        batch_size=NUM_IMAGES * 2,
    )
    batch = next(iter(testloader))[0].to(DEVICE)

    # batch = torch.randn((128, 3, 128, 128)).to(DEVICE)

    a = model(batch[:NUM_IMAGES])[2]
    b = model(batch[NUM_IMAGES:])[2]

    print("Interpolating discretely...")
    trajectory = simple_interpolate(a, b, steps=NUM_STEPS, codebook=model.codebook.data)

    print(trajectory[0].shape)

    print("Creating GIF...")
    animate_interpolation(trajectory, model.decoder, "testanim")
