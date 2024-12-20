"Python script to train different models"

from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, Optimizer
from torchvision.transforms import v2
import yaml
from tqdm.auto import tqdm
from pml_vqvae.stats_keeper import StatsKeeper
from pml_vqvae.wandb_wrapper import WANDBWrapper
from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.cli_handler import CLI_handler
from pml_vqvae.train_config import TrainConfig
from pml_vqvae.dataset.dataloader import load_data

# import wandb
DEFAULT_CONFIG = "config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test(
    model: PML_model,
    test_loader: DataLoader,
    stats_keeper: StatsKeeper,
    label_conditioning: bool,
):
    """Test a model on a dataset

    Args:
        model (PML_model): The model to test
        test_loader (DataLoader): The data loader to use
        stats_keeper (StatsKeeper): The stats keeper to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The last batch, target and output
    """

    losses = []

    model.eval()

    # for dynamic logging
    test_tqdm = tqdm(test_loader)

    for batch, labels in test_tqdm:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        output = model(batch, labels) if label_conditioning else model(batch)
        loss = model.loss_fn(output, batch)
        losses.append(loss.item())

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(model.batch_stats, len(batch), train=False)

        # make a nice progress bar
        test_tqdm.set_description(dsp)

    # create epoch level stats
    stats_keeper.batch_summarize(train=False)

    model.train()

    avg_loss = sum(losses) / len(losses)

    return batch, output, avg_loss


def train_epoch(
    model: PML_model,
    train_loader: DataLoader,
    optimizer: Optimizer,
    stats_keeper: StatsKeeper,
    label_conditioning: bool,
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

    for batch, labels in train_tqdm:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        output = model(batch, labels) if label_conditioning else model(batch)
        loss = model.loss_fn(output, batch)
        model.backward(loss)

        # collect all stats in Object for later plotting
        dsp = stats_keeper.add_batch_stats(model.batch_stats, len(batch))

        # make a nice progress bar
        train_tqdm.set_description(dsp)

        optimizer.step()

    # create epoch level stats
    stats_keeper.batch_summarize()

    return batch, output, loss.item()


def train(config: TrainConfig):
    """Train a model on a dataset for a number of epochs

    Args:
        train_config (TrainConfig): Training configuration

    Returns:
        float: Some value that quantizes the generalization error, the lower the better.
    """

    model = config.get_model()

    print(
        f"Training {config.model_name} on "
        f"{config.dataset} for {config.epochs} epochs"
    )

    print("Loading dataset...")

    train_loader, test_loader = load_data(
        config.dataset,
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

    last_average_test_loss = None

    print("Training model...")
    for i in range(config.epochs):
        # train on all datat for one epoch
        batch, output, _ = train_epoch(
            model,
            train_loader,
            optimizer,
            stats_keeper,
            config.label_conditioning,
        )
        print(f"Batch images are in range [{batch.min()}, {batch.max()}]")
        wandb_wrapper.construct_examples(batch, model.visualize_output(output))

        # test
        if (
            config.test_interval and i % config.test_interval == 0
        ) or i == config.epochs - 1:
            with torch.no_grad():
                batch, output, last_average_test_loss = test(
                    model,
                    test_loader,
                    stats_keeper,
                    config.label_conditioning,
                )
                wandb_wrapper.construct_examples(
                    batch, model.visualize_output(output), train=False
                )

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

    return last_average_test_loss


# Ich habe die CLI functionality auskommentiert um es mir einfache zu machen den
# stuff für die Hyperparameteroptimisierung zu integrieren, sorry für
# unsaubere Arbeit.
if __name__ == "__main__":
    # cli_handler = CLI_handler()
    # args = cli_handler.parse_args()

    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as file:
        config = TrainConfig.from_dict(yaml.safe_load(file))

    # # Overwrite config when cli arguments are provided
    # config = cli_handler.adjust_config(config, args)

    print(f"Starting training with the following onconfiguration:\n\n{config}\n")

    train(config)
