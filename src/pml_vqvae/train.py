"Python script to train different models"

import argparse
from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import v2
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    loss_fn = model.loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(DEVICE)

    # Train model
    losses = []  # losses over epochs
    vals = []
    for i in range(epochs):
        batch_losses = []
        # iterate over batches
        for batch_img, _ in train_loader:
            batch_img = batch_img.to(DEVICE)

            optimizer.zero_grad()

            # returns a tuple of losses: for basic autoencoder it's just the reconstruction, but for VAE it's (reconstruction, mean, logvar)
            output = model(batch_img)

            # In case of VAE, output of model is a tuple of (reconstruction, mean, logvar)
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
    torch.save(model.state_dict(), "testmodel_5ep.pth")
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
