import torch
import yaml
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.dataset.latent import LatentDatasetGenerator
from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.cli_handler import CLI_handler
import argparse
from torchvision.transforms import v2


def generate_latent_dataset(
    data_loader: torch.utils.data.DataLoader,
    vqvae: VQVAE,
) -> LatentDatasetGenerator:
    vqvae.eval()

    latent_dataset = LatentDatasetGenerator()

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
        "--ns",
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
        config_file = f"{args['model_path']}/config.yaml"
        model_file = f"{args['model_path']}/model.pth"

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        model_config = VQVAEConfig(**config_dict["model_config"])

        vqvae = VQVAE(model_config)
        vqvae.load_state_dict(torch.load(model_file, weights_only=True))

    except Exception as e:
        print(f"Could not load model from {args.model_path}")
        print(e)
        exit()

    dataset = args.dataset
    n_samples = args.n_samples
    seed = args.seed

    train_loader, test_loader = load_data(
        dataset,
        n_train=n_samples,
        n_test=None,
        seed=seed,
        class_idx=None,
        batch_size=128,
    )

    train_latent_dataset = generate_latent_dataset(train_loader, vqvae)
    train_latent_dataset.save(f"{dataset}_latents_{n_samples}/train", "train")

    test_latent_dataset = generate_latent_dataset(test_loader, vqvae)
    test_latent_dataset.save(f"{dataset}_latents_{n_samples}/test", "test")
