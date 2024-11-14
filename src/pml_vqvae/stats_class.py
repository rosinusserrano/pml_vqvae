import matplotlib.pyplot as plt
import wandb
import os
import torch
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.config_class import Config
import numpy as np


class StatsKeeper:
    """Class to keep track of stats during training and testing

    Args:
        output_dir (str, optional): The output directory to save the plots to. Defaults to None.

    Raises:
        ValueError: If output_dir is not set
    """

    def __init__(self, config: Config, output_dir: str = None):

        self.output_dir = output_dir

        self.train_epoch_stats = {}
        self.train_batch_stats = {}

        self.test_epoch_stats = {}
        self.test_batch_stats = {}

        self.wandb_log = config.wandb_log

        self.epoch_cnt = 0

        if config.wandb_log:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project="pml_vqvae", name=config.name, config=config.to_dict())
            # wandb.watch(model, log_freq=1)

    def add_batch_stats(self, stats: dict, train: bool = True):
        """Add stats for one batch

        Args:
            stats (dict): The stats to add
            train (bool, optional): Whether this is a training batch. Defaults to True.

        Returns:
            str: A string representation of the stats
        """
        for key, value in stats.items():
            if train:
                self.train_batch_stats.setdefault(key, []).append(value)
            else:
                self.test_batch_stats.setdefault(key, []).append(value)

        return f"[{'train' if train else 'test'}] " + " | ".join(
            [f"{k}:{v:.2f}" for k, v in stats.items()]
        )

    def batch_summarize(self, train: bool = True):
        """Summarize the stats for the epoch

        Args:
            train (bool, optional): Whether this is a training epoch. Defaults to True.
        """

        if train:
            train_epoch_stats = {
                k: sum(v) / len(v) for k, v in self.train_batch_stats.items()
            }

            for key, value in train_epoch_stats.items():
                self.train_epoch_stats.setdefault(key, []).append(value)

            if self.wandb_log:
                train_epoch_stats = {
                    "train/" + k: v for k, v in train_epoch_stats.items()
                }
                wandb.log(train_epoch_stats, step=self.epoch_cnt)

            self.epoch_cnt += 1
            self.train_batch_stats = {}
        else:
            test_epoch_stats = {
                k: sum(v) / len(v) for k, v in self.test_batch_stats.items()
            }

            for key, value in test_epoch_stats.items():
                self.test_epoch_stats.setdefault(key, []).append(value)

            if self.wandb_log:
                test_epoch_stats = {"test/" + k: v for k, v in test_epoch_stats.items()}
                wandb.log(test_epoch_stats, step=self.epoch_cnt)

            self.test_batch_stats = {}

    def save_examples(self, batch, output, epoch: int):
        if self.wandb_log:
            payload = {
                "examples": [
                    wandb.Image(np.moveaxis(batch[i].cpu().detach().numpy(), 0, -1))
                    for i in range(len(batch))
                ],
                "reconstructions": [
                    wandb.Image(np.moveaxis(output[i].cpu().detach().numpy(), 0, -1))
                    for i in range(len(batch))
                ],
            }
            wandb.log(payload, step=epoch)

    def save_model(self, model: PML_model, epoch: int):
        """Save the model to the output directory

        Args:
            model (PML_model): The model to save
            epoch (int): The current epoch
        """
        if self.output_dir is None:
            raise ValueError("Output directory is not set")

        name = f"{self.output_dir}/model_{epoch}.pth"
        torch.save(model.state_dict(), name)

        if self.wandb_log:
            wandb.save(name)

    def finish(self):
        """Finish the logging process"""
        if self.wandb_log:
            wandb.finish()
        else:
            self.visualize()

    def visualize(self):
        """Visualize the stats and save the plots to the output directory"""
        if self.output_dir is None:
            raise ValueError("Output directory is not set")

        for stat in self.train_epoch_stats.keys():
            plt.clf()
            plt.plot(self.train_epoch_stats[stat])
            plt.plot(self.test_epoch_stats[stat])
            plt.title(stat)
            plt.savefig(f"{self.output_dir}/plots/{stat}.png")
