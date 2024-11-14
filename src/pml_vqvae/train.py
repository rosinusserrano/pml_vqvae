"Python script to train different models"

import os
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.cli_input_handler import CLI_handler
from pml_vqvae.config_class import Config
from pml_vqvae.dataset.dataloader import load_data
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, Optimizer
from torchvision.transforms import v2
import yaml
from tqdm.auto import tqdm

from pml_vqvae.stats_class import StatsKeeper

# import wandb
DEFAULT_CONFIG = "config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# From wandb tutotial https://docs.wandb.ai/tutorials/pytorch
def train_log(loss, epoch, epochs):
    # Where the magic happens
    # wandb.log({"epoch": epoch, "loss": loss}, step=epoch)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")


def test(model: PML_model, test_loader: DataLoader, stats_keeper: StatsKeeper):
    """Test a model on a dataset

    Args:
        model (PML_model): The model to test
        test_loader (DataLoader): The data loader to use
        stats_keeper (StatsKeeper): The stats keeper to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The last batch, target and output
    """

    model.eval()

    # for dynamic logging
    test_tqdm = tqdm(test_loader)

    for batch, target in test_tqdm:
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)

        output = model(batch)
        loss = model.loss_fn(output, target)

        # collect model specific stats
        stats = model.collect_stats(output, target, loss)

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(stats, train=False)

        # make a nice progress bar
        test_tqdm.set_description(dsp)

    # create epoch level stats
    stats_keeper.batch_summarize(train=False)

    model.train()

    return batch, target, output


def train_epoch(
    model: PML_model,
    train_loader: DataLoader,
    optimizer: Optimizer,
    stats_keeper: StatsKeeper,
):
    """Train a model on a dataset for one epoch

    Args:
        model (PML_model): The model to train
        train_loader (DataLoader): The data loader to use
        optimizer (Optimizer): The optimizer to use
        stats_keeper (StatsKeeper): The stats keeper to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The last batch, target and output
    """

    # for dynamic logging
    train_tqdm = tqdm(train_loader)

    for batch, target in train_tqdm:
        batch = batch.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        output = model(batch)
        loss = model.loss_fn(output, target)
        model.backward(loss)

        # collect model specific stats
        stats = model.collect_stats(output, target, loss)

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(stats)

        # make a nice progress bar
        train_tqdm.set_description(dsp)

        optimizer.step()

    # create epoch level stats
    stats_keeper.batch_summarize()

    return batch, target, output


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

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.to(DEVICE)

    print("Training model...")
    stats_keeper = StatsKeeper(output_dir=config.output_dir)

    for i in range(config.epochs):
        # train on all datat for one epoch
        batch, target, output = train_epoch(
            model, train_loader, optimizer, stats_keeper
        )

        if i % 1 == 0:
            model.visualize_output(
                batch,
                output,
                target,
                prefix=f"train_epoch{i}",
                base_dir=os.path.join(config.output_dir, "visuals"),
            )

        # test on all data for one epoch
        with torch.no_grad():
            batch, target, output = test(model, test_loader, stats_keeper)

            if i % 1 == 0:
                model.visualize_output(
                    batch,
                    output,
                    target,
                    prefix=f"test_epoch{i}",
                    base_dir=os.path.join(config.output_dir, "visuals"),
                )

    print("Saving model...")
    torch.save(
        model.state_dict(),
        os.path.join(config.output_dir, "models", f"{config.model_name}.pth"),
    )

    print("Plotting stats...")
    stats_keeper.visualize()


cli_handler = CLI_handler()
args = cli_handler.parse_args()

with open(DEFAULT_CONFIG, "r") as file:
    config = Config.from_dict(yaml.safe_load(file))

# Overwrite config when cli arguments are provided
config = cli_handler.adjust_config(config, args)

print(config)

train(config)
