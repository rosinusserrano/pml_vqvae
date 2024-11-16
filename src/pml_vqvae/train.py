"Python script to train different models"

import os
import matplotlib.pyplot as plt
from pml_vqvae.cli_input_handler import CLI_handler
from pml_vqvae.config_class import Config
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import v2
import yaml
from tqdm.auto import tqdm

# import wandb
DEFAULT_CONFIG = "config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("artifacts/models/", exist_ok=True)
os.makedirs("artifacts/visuals/", exist_ok=True)
os.makedirs("artifacts/plots/", exist_ok=True)


# From wandb tutotial https://docs.wandb.ai/tutorials/pytorch
def train_log(loss, epoch, epochs):
    # Where the magic happens
    # wandb.log({"epoch": epoch, "loss": loss}, step=epoch)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")


def train(config: Config):
    """Train a model on a dataset for a number of epochs

    Args:
        config (dict): Configuration dictionary

    Returns:
        np.array: The losses over epochs (either a list of multiple losses or a list of floats)
    """

    model = config.get_model()

    print(f"Training {model.name()} on {config.dataset} for {config.epochs} epochs")

    # Load data
    print("Loading dataset...")
    # stolen from https://pytorch.org/vision/main/transforms.html
    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ]
    )
    train_loader, test_loader = load_data(
        config.dataset,
        transformation=transforms,
        n_train=config.n_train,
        n_test=config.n_test,
        seed=config.seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.to(DEVICE)

    print("Training model...")
    train_epoch_stats = {}
    test_epoch_stats = {}

    for i in range(config.epochs):
        train_batch_stats = {}
        train_tqdm = tqdm(train_loader)
        for batch, target in train_tqdm:
            batch = batch.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            output = model(batch)
            loss = model.loss_fn(output, target)
            model.backward(loss)

            stats = model.collect_stats(output, target, loss)
            for key, value in stats.items():
                train_batch_stats.setdefault(key, []).append(value)

            train_tqdm.set_description(
                "[train] " + " | ".join([f"{k}:{v:.2f}" for k, v in stats.items()])
            )

            optimizer.step()

        if i % 1 == 0:
            model.visualize_output(
                batch,
                output,
                target,
                prefix=f"train_epoch{i}",
                base_dir=f"{config.output_dir}/visuals",
            )

        train_batch_stats = {k: sum(v) / len(v) for k, v in train_batch_stats.items()}
        for key, value in train_batch_stats.items():
            train_epoch_stats.setdefault(key, []).append(value)

        with torch.no_grad():
            model.eval()

            test_batch_stats = {}
            test_tqdm = tqdm(test_loader)

            for batch, target in test_tqdm:
                batch = batch.to(DEVICE)
                target = target.to(DEVICE)

                output = model(batch)
                loss = model.loss_fn(output, target)

                stats = model.collect_stats(output, target, loss)
                for key, value in stats.items():
                    test_batch_stats.setdefault(key, []).append(value)

                test_tqdm.set_description(
                    "[test] " + " | ".join([f"{k}:{v:.2f}" for k, v in stats.items()])
                )

            model.train()

            if i % 1 == 0:
                model.visualize_output(
                    batch,
                    output,
                    target,
                    prefix=f"test_epoch{i}",
                    base_dir=f"{config.output_dir}/visuals",
                )

            test_batch_stats = {k: sum(v) / len(v) for k, v in test_batch_stats.items()}
            for key, value in test_batch_stats.items():
                test_epoch_stats.setdefault(key, []).append(value)

    print("Saving model...")
    torch.save(model.state_dict(), f"{config.output_dir}/models/testmodel_5ep.pth")

    print("Plotting stats...")
    for stat in train_epoch_stats.keys():
        plt.clf()
        plt.plot(train_epoch_stats[stat])
        plt.plot(test_epoch_stats[stat])
        plt.title(stat)
        plt.savefig(f"{config.output_dir}/plots/{stat}.png")


cli_handler = CLI_handler()
args = cli_handler.parse_args()

with open(DEFAULT_CONFIG, "r") as file:
    config = Config.from_dict(yaml.safe_load(file))

# Overwrite config when cli arguments are provided
config = cli_handler.adjust_config(config, args)

print(config)

losses = train(config)
print(losses)
