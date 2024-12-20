import torch
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.dataset.latent import LatentDataset
from pml_vqvae.models.vqvae import VQVAE
from pml_vqvae.cli_handler import CLI_handler
import argparse
from torchvision.transforms import v2


def generate_latent_dataset(
    data_loader: torch.utils.data.DataLoader,
    vqvae: VQVAE,
) -> LatentDataset:
    vqvae.eval()

    latent_dataset = LatentDataset()

    for batch, labels in data_loader:

        # run model o batch
        vqvae.vqvae(batch)

        # get batch latents
        b_latent = vqvae.discrete_latent.reshape(-1, 32, 32)

        # add latents to dataset
        latent_dataset.add_latent(b_latent.cpu().numpy(), labels.cpu().numpy())

    return latent_dataset


if __name__ == "__main__":
    cli_handler = CLI_handler()
    args = cli_handler.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        help="Path to the vqvae model pth-file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--n_samples",
        "-ns",
        help="Number of samples to use",
        type=int,
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="Seed for reproducibility",
        type=int,
    )

    args = parser.parse_args()

    try:
        vqvae = VQVAE.load_from_checkpoint(args.model_path, weights_only=True)
    except Exception as e:
        print(f"Could not load model from {args.model_path}")
        print(e)
        exit()

    dataset = args.dataset
    n_samples = args.n_samples
    seed = args.seed

    transforms = (
        v2.Compose(
            [
                v2.RandomResizedCrop(size=(128, 128), antialias=True, scale=(0.1, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if dataset == "imagenet"
        else v2.Compose(
            [
                v2.RandomResizedCrop(size=(32, 32), antialias=True, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
            ]
        )
    )

    train_loader, _ = load_data(
        dataset,
        transformation=transforms,
        n_train=n_samples,
        n_test=None,
        seed=seed,
        class_idx=None,
        batch_size=64,
    )

    latent_dataset = generate_latent_dataset(train_loader, vqvae)

    latent_dataset.save(f"{dataset}_latents_{n_samples}.npy")
