from tqdm.auto import tqdm
import torch
from torch.nn.functional import mse_loss
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.models.vqvae import VQVAEConfig, VQVAE

from skimage.metrics import structural_similarity as ssim

import yaml
import pickle


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("On device", DEVICE)


_, imagenet30classes_test = load_data("imagenet", batch_size=128, class_idx=range(30))

vqvae_config = VQVAEConfig(
    codebook_size=1024,
    commitment_weight=2.2348288297357928,
    hidden_dimension=64,
    embedding_dimension=64,
)

vqvae = VQVAE(vqvae_config)
vqvae.load_state_dict(
    torch.load(
        "artifacts/vqvae_konni_easy_params/model.pth",
        weights_only=True,
    )
)

losses_per_class = {}

for batch, label in tqdm(imagenet30classes_test):
    batch.to(DEVICE)
    label.to(DEVICE)

    reconstructions, _, _, _ = vqvae(batch)

    loss_per_sample = mse_loss(reconstructions, batch, reduction="none").mean(
        dim=(1, 2, 3)
    )

    ssim_losses = []
    for i in range(reconstructions.shape[0]):
        ssim_losses.append(
            ssim(
                reconstructions[i].detach().cpu().numpy(),
                batch[i].detach().cpu().numpy(),
                channel_axis=0,
                data_range=(batch[i].max() - batch[i].min()).item(),
            )
        )

    for i in range(reconstructions.shape[0]):
        losses_per_class.setdefault(label[i].item(), {}).setdefault("mse", []).append(
            loss_per_sample[i].item()
        )
        losses_per_class.setdefault(label[i].item(), {}).setdefault("ssim", []).append(
            ssim_losses[i].item()
        )

print("Done iterating through dataset")

with open("losses_per_class.yaml", "w") as f:
    print("Writing yaml")
    yaml.safe_dump(losses_per_class, f)

with open("losses_per_class.pkl", "wb") as f:
    print("Writing pickle")
    pickle.dump(losses_per_class, f)
