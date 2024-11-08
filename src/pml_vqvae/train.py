"Python script to train different models"

import argparse
from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import RandomCrop

# import wandb

# Hyperparameters
MODEL = BaselineAutoencoder()
DATASET = "cifar"

EPOCHS = 100
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


def train(model: torch.nn.Module = MODEL, dataset: str = DATASET, epochs: int = EPOCHS):
    print(f"Training {model.name()} on {dataset} for {epochs} epochs")

    # Load data
    print("Loading dataset")
    train_loader, test_loader = load_data(
        dataset,
        RandomCrop(128, pad_if_needed=True),
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

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

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

            output = model(batch_img)
            loss = loss_fn(output, batch_img)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(epoch_loss)
        train_log(epoch_loss, i, epochs)

    return losses


parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model you want to train")
parser.add_argument("dataset", help="The dataset onto which to train your model")

args = parser.parse_args()

assert args.model in ["vae", "autoencoder"], "Unknown model"
assert args.dataset in ["cifar", "imagenet"], "Unknown dataset"

model = BaselineAutoencoder() if args.model == "autoencoder" else None
losses = train(model=model, dataset=args.dataset)
