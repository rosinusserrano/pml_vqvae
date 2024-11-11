"Python script to train different models"

import argparse
from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import RandomCrop
import numpy as np

# import wandb

# Hyperparameters
MODEL = BaselineAutoencoder()
DATASET = "cifar"

EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9
N_TRAIN = 1000
N_TEST = 1000

SEED = 2024


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
    print("Loading dataset")

    # TODO: Transformations should be added here
    train_loader, test_loader = load_data(
        dataset,
        transformation=RandomCrop(128, pad_if_needed=True),
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=SEED,
    )

    # wandb.init(
    #     project="PML-VQVAE",
    #     config={
    #         "learning_rate": LEARNING_RATE,
    #         "architecture": model.name(),
    #         "dataset": dataset,
    #         "n_train": N_TRAIN,
    #         "n_test": N_TEST,
    #         "epochs": epochs,
    #     },
    # )

    loss_fn = model.loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Watch model with wandb
    # wandb.watch(model, loss_fn, log="all", log_freq=10)

    # Train model
    losses = []  # losses over epochs
    vals = []
    for i in range(epochs):
        batch_losses = []
        # iterate over batches
        for batch_img, _ in train_loader:
            optimizer.zero_grad()

            # returns a tuple of losses: for basic autoencoder it's just the reconstruction, but for VAE it's (reconstruction, mean, logvar)
            output = model(batch_img)

            # In case of VAE, loss is a tuple of (loss, mean, logvar)
            loss = loss_fn(*output, batch_img)

            # whether loss is a tensor (autoencoder) or a tuple (VAE)
            if isinstance(loss, torch.Tensor):
                model.backward(loss)
                batch_losses.append(loss.item())
            elif isinstance(loss, tuple):
                model.backward(loss[0])
                batch_losses.append([l.item() for l in loss])
            optimizer.step()

        # claculate mean loss over all batches
        epoch_loss = np.array(batch_losses).mean(axis=0)
        losses.append(epoch_loss)
        train_log(epoch_loss, i, epochs)

    # losses can be either a list of floats (autoencoder) or a list of lists (VAE)
    return np.array(losses)


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

losses = train(model=model, dataset=args.dataset)
print(losses)
