"Python script to train different models"

from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, Optimizer
from torchvision.transforms import v2
import yaml
from tqdm.auto import tqdm
from pml_vqvae.stats_keeper import StatsKeeper
from pml_vqvae.wandb_wrapper import WANDBWrapper
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.cli_handler import CLI_handler
from pml_vqvae.train_config import TrainConfig
from pml_vqvae.dataset.dataloader import load_data

# import wandb
DEFAULT_CONFIG = "config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(model.batch_stats, len(batch), train=False)

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

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(model.batch_stats, len(batch))

        # make a nice progress bar
        train_tqdm.set_description(dsp)

        optimizer.step()

    # create epoch level stats
    stats_keeper.batch_summarize()

    return batch, target, output


def train(config: TrainConfig):
    """Train a model on a dataset for a number of epochs

    Args:
        config (dict): Configuration dictionary

    Returns:
        np.array: The losses over epochs (either a list of multiple losses or a list of floats)
    """

    model = config.get_model()

    print(f"Training {model.name()} on {config.dataset} for {config.epochs} epochs")

    print("Loading dataset...")

    transforms = (
        v2.Compose(
            [
                v2.RandomResizedCrop(size=(128, 128), antialias=True, scale=(0.1, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if config.dataset == "imagenet"
        else v2.Compose(
            [
                v2.RandomResizedCrop(size=(32, 32), antialias=True, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
            ]
        )
    )

    train_loader, test_loader = load_data(
        config.dataset,
        transformation=transforms,
        n_train=config.n_train,
        n_test=config.n_test,
        seed=config.seed,
        class_idx=config.class_idx,
        batch_size=config.batch_size,
    )

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.to(DEVICE)

    wandb_wrapper = WANDBWrapper(config)
    wandb_wrapper.init(model)

    stats_keeper = StatsKeeper()

    print("Training model...")
    for i in range(config.epochs):
        # train on all datat for one epoch
        batch, _, output = train_epoch(model, train_loader, optimizer, stats_keeper)
        wandb_wrapper.construct_examples(batch, output)

        # test
        if config.test_interval and i % config.test_interval == 0:
            with torch.no_grad():
                batch, _, output = test(model, test_loader, stats_keeper)
                wandb_wrapper.construct_examples(batch, output, train=False)

        log_vis = True
        if not config.vis_train_interval or i % config.vis_train_interval != 0:
            log_vis = False

        epoch_stats = stats_keeper.get_latest_stats()
        wandb_wrapper.log_epoch(epoch_stats, epoch=i, log_vis=log_vis)

        model_dir = stats_keeper.save_model(model, config.output_dir, epoch=i)
        wandb_wrapper.save_model(model_dir)

    # save final model
    print("Saving model...")
    stats_keeper.save_model(model, config.output_dir, epoch=config.epochs)
    stats_keeper.plot_results(config.output_dir)
    wandb_wrapper.save_model(model_dir)
    wandb_wrapper.finish()


cli_handler = CLI_handler()
args = cli_handler.parse_args()

with open(DEFAULT_CONFIG, "r", encoding="utf-8") as file:
    config = TrainConfig.from_dict(yaml.safe_load(file))

# Overwrite config when cli arguments are provided
config = cli_handler.adjust_config(config, args)

print(f"Starting training with the following onconfiguration:\n\n{config}\n")

train(config)
