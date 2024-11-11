"Python script to train different models"

import os
import argparse
import matplotlib.pyplot as plt
from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import v2
import numpy as np
from tqdm.auto import tqdm

# import wandb

# Hyperparameters
MODEL = BaselineAutoencoder()
DATASET = "cifar"

EPOCHS = 200
LEARNING_RATE = 0.01
MOMENTUM = 0.9
N_TRAIN = 1000
N_TEST = 1000

SEED = 2024


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("artifacts/models/", exist_ok=True)
os.makedirs("artifacts/visuals/", exist_ok=True)
os.makedirs("artifacts/plots/", exist_ok=True)


# From wandb tutotial https://docs.wandb.ai/tutorials/pytorch
def train_log(loss, epoch, epochs):
    # Where the magic happens
    # wandb.log({"epoch": epoch, "loss": loss}, step=epoch)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")


def train(model: PML_model = MODEL, dataset: str = DATASET, epochs: int = EPOCHS):
    """Train a model on a dataset for a number of epochs

    Args:
        model (PML_model, optional): The model to train. Defaults to MODEL.
        dataset (str, optional): The dataset to train on. Defaults to DATASET.
        epochs (int, optional): The number of epochs to train for. Defaults to EPOCHS.

    Returns:
        np.array: The losses over epochs (either a list of multiple losses or a list of floats)
    """

    print(f"Training {model.name()} on {dataset} for {epochs} epochs")

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
        dataset,
        batch_size=32,
        transformation=transforms,
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=SEED,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(DEVICE)

    print("Training model...")
    train_epoch_stats = {}
    test_epoch_stats = {}

    for i in range(epochs):
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
                base_dir="artifacts/visuals",
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
                    base_dir="artifacts/visuals",
                )

            test_batch_stats = {k: sum(v) / len(v) for k, v in test_batch_stats.items()}
            for key, value in test_batch_stats.items():
                test_epoch_stats.setdefault(key, []).append(value)

    print("Saving model...")
    torch.save(model.state_dict(), "artifacts/models/testmodel_5ep.pth")

    print("Plotting stats...")
    for stat in train_epoch_stats.keys():
        plt.clf()
        plt.plot(train_epoch_stats[stat])
        plt.plot(test_epoch_stats[stat])
        plt.title(stat)
        plt.savefig(f"artifacts/plots/{stat}.png")


parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model you want to train")
parser.add_argument("dataset", help="The dataset onto which to train your model")

args = parser.parse_args()

assert args.model in ["vae", "autoencoder"], "Unknown model"
assert args.dataset in ["cifar", "imagenet"], "Unknown dataset"

model = None
if args.model == "autoencoder":
    model = BaselineAutoencoder()
elif args.model == "vae":
    model = BaselineVariationalAutoencoder()

train(model=model, dataset=args.dataset)
