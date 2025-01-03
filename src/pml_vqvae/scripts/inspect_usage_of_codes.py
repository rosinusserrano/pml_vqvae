import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from pml_vqvae.dataset.dataloader import load_data


def get_usage_of_codes(train: bool = True, outfile: str | None = None):
    train_loader, test_loader = load_data("latent")
    dataloader = train_loader if train else test_loader

    bincount = torch.zeros((1024,))
    for batch, labels in tqdm(dataloader):
        bincount += torch.bincount(batch.flatten().long(), minlength=1024)

    usage_in_percent = bincount / bincount.sum()

    xticks = [str(i) if usage_in_percent[i].item() > 0 else "" for i in range(1024)]

    number_used_codes = len([i for i in range(1024) if usage_in_percent[i] > 0])

    if outfile is not None:
        plt.figure(figsize=(20, 12))
        plt.bar(
            xticks,
            usage_in_percent.detach().cpu().numpy(),
        )
        plt.xticks(rotation=90)
        plt.ylabel("Code usage (%)")
        plt.xlabel("Code index")
        plt.savefig(outfile)

        plt.clf()

        plt.figure(figsize=(10, 5))
        plt.barh(y=[""], width=number_used_codes / 1024, height=0.8, label="Used codes")
        plt.barh(
            [""],
            (1024 - number_used_codes) / 1024,
            0.8,
            left=number_used_codes / 1024,
            label="Ignored codes",
        )
        plt.yticks([])
        plt.xlabel("Code usage (%)")
        plt.legend()
        plt.savefig(f"compare_{outfile}")

    return bincount


if __name__ == "__main__":
    get_usage_of_codes(train=False, outfile="usage_of_codes.svg")
