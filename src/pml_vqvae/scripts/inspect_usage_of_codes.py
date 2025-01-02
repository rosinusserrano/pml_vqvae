import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from pml_vqvae.dataset.dataloader import load_data


def get_usage_of_codes():
    train_loader, _ = load_data("latent")

    bincount = torch.zeros((1024,))

    for batch, labels in tqdm(train_loader):
        bincount += torch.bincount(batch.flatten().long(), minlength=1024)

    # plt.scatter(range(1024), bincount.detach().cpu().numpy())
    # plt.savefig("test.png")

    return bincount
