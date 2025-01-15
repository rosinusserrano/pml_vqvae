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
    max_per_file: int = 1,
) -> LatentDatasetGenerator:
    vqvae.eval()

    latent_dataset = LatentDatasetGenerator(max_per_file=max_per_file)

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
        required=True
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Name of the dataset to use",
        required=True
    )
    parser.add_argument(
        "--n-train",
        "--ntr",
        help="Number of training samples to use",
        type=int,
    )
    parser.add_argument(
        "--n-test",
        "--nte",
        help="Number of test samples to use",
        type=int,
    )
    parser.add_argument(
        "--n-classes",
        "--nc",
        help="Number of classes to use",
        type=int,
    )
    parser.add_argument(
        "--max-per-file",
        "--mpf",
        help="Number of samples packed into a single file",
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
    n_train = args.n_train
    n_test = args.n_test
    n_classes = args.n_classes
    max_per_file = args.max_per_file
    seed = args.seed

    if (n_train is not None or n_test is not None) and n_classes is not None:
        raise ValueError("Either use --n-train/--n-test or --n-classes, not mixed.")
    
    dataset_name = f"{dataset}_latents"
    if n_classes is not None:
        dataset_name = f"{dataset_name}_{n_classes}classes"
    if n_train is not None:
        dataset_name = f"{dataset_name}_{n_train}train"
    if n_test is not None:
        dataset_name = f"{dataset_name}_{n_test}test"

    print("Loading data")
    train_loader, test_loader = load_data(
        dataset,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        class_idx=list(range(n_classes)) if n_classes is not None else None,
        batch_size=256,
        shuffle=False,
    )

    print("Iterating through train set")
    train_latent_dataset = generate_latent_dataset(train_loader, vqvae, max_per_file)
    print("Saving to filesystem")
    train_latent_dataset.save(
        f"{args.model_path}/{dataset_name}/train", "train"
    )

    print("Iterating through test set")
    test_latent_dataset = generate_latent_dataset(test_loader, vqvae, max_per_file)
    print("Saving to filesystem")
    test_latent_dataset.save(
        f"{args.model_path}/{dataset_name}/test", "test"
    )
