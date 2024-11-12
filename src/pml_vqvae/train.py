"Python script to train different models"

from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.cli_input_handler import CLI_handler
from pml_vqvae.config_class import Config
from pml_vqvae.dataset.dataloader import load_data
import torch
from torchvision.transforms import RandomCrop
import numpy as np
import yaml

DEFAULT_CONFIG = "config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    print("Loading dataset")

    # TODO: Transformations should be added here
    train_loader, test_loader = load_data(
        config.dataset,
        transformation=RandomCrop(128, pad_if_needed=True),
        n_train=config.n_train,
        n_test=config.n_test,
        seed=config.seed,
    )

    loss_fn = model.loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.to(DEVICE)

    # Train model
    losses = []  # losses over epochs
    vals = []
    for i in range(config.epochs):
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
        train_log(epoch_loss, i, config.epochs)

    torch.save(model.state_dict(), "testmodel_5ep.pth")
    # losses can be either a list of floats (autoencoder) or a list of lists (VAE)
    return np.array(losses)


cli_handler = CLI_handler()
args = cli_handler.parse_args()

with open(DEFAULT_CONFIG, "r") as file:
    config = Config.from_dict(yaml.safe_load(file))

# Overwrite config when cli arguments are provided
config = cli_handler.adjust_config(config, args)

print(config)

losses = train(config)
print(losses)
