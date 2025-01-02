import torch
import yaml
from tqdm.auto import tqdm
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.dataset.latent import LatentDatasetGenerator
from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.cli_handler import CLI_handler
import argparse
from torchvision.transforms import v2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"On device: {DEVICE}")


def generate_latent_dataset(
    data_loader: torch.utils.data.DataLoader,
    vqvae: VQVAE,
) -> LatentDatasetGenerator:
    vqvae.eval()

    latent_dataset = LatentDatasetGenerator()

    vqvae.to(DEVICE)

    for batch, labels in tqdm(data_loader):
        code_indices = vqvae.encode(batch.to(DEVICE))
        latent_dataset.add_latent(code_indices.cpu().numpy(), labels.cpu().numpy())

    return latent_dataset


if __name__ == "__main__":
    # cli_handler = CLI_handler()
    # args = cli_handler.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        help="Path to a directory containing config.yaml and model.pth",
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
        config_file = f"{args.model_path}/config.yaml"
        model_file = f"{args.model_path}/model.pth"

        print("Reading config file")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        print("Loading model")
        model_config = VQVAEConfig(**config_dict["model_config"]["value"])
        vqvae = VQVAE(model_config)
        vqvae.load_state_dict(torch.load(model_file, weights_only=True))

    except Exception as e:
        print(f"Could not load model from {args.model_path}")
        print(e)
        exit()

    dataset = args.dataset
    n_samples = args.n_samples
    seed = args.seed

    print("Loading data")
    train_loader, test_loader = load_data(
        dataset,
        n_train=None,
        n_test=None,
        seed=seed,
        class_idx=list(range(30)),
        batch_size=256,
    )

    print("Generating train set")
    train_latent_dataset = generate_latent_dataset(train_loader, vqvae)
    train_latent_dataset.save(
        f"artifacts/30_classes_{dataset}_latents_{n_samples}/train", "train"
    )

    print("Generating test set")
    test_latent_dataset = generate_latent_dataset(test_loader, vqvae)
    test_latent_dataset.save(
        f"artifacts/30_classes_{dataset}_latents_{n_samples}/test", "test"
    )
