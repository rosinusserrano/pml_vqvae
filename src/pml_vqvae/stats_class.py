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

    def __init__(self):

        self.train_epoch_stats = {}
        self.train_batch_stats = {}

        self.test_epoch_stats = {}
        self.test_batch_stats = {}

        self.example_cnt = 0

    def save_model(self, model: PML_model, output_dir: str, epoch: int):
        """Save the model to the output directory

        Args:
            model (PML_model): The model to save
            epoch (int): The current epoch
        """

        name = f"{output_dir}/model_{epoch}.pth"
        torch.save(model.state_dict(), name)

        return name

    def get_latest_stats(self):
        """Get the latest stats

        Returns:
            Tuple[dict, dict]: The latest train and test stats
        """

        try:
            train_stats = {k: v[-1] for k, v in self.train_epoch_stats.items()}
        except IndexError:
            test_stats = {}

        try:
            test_stats = {k: v[-1] for k, v in self.test_epoch_stats.items()}
        except IndexError:
            test_stats = {}

        return train_stats, test_stats, self.example_cnt

    def add_batch_stats(self, stats: dict, batch_size: int, train: bool = True):
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

        if train:
            self.example_cnt += batch_size

        return f"[{'train' if train else 'test'}] " + " | ".join(
            [f"{k}: {v:.2f}" for k, v in stats.items()]
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

            self.train_batch_stats = {}
        else:
            test_epoch_stats = {
                k: sum(v) / len(v) for k, v in self.test_batch_stats.items()
            }

            for key, value in test_epoch_stats.items():
                self.test_epoch_stats.setdefault(key, []).append(value)

            self.test_batch_stats = {}

    def plot_results(self, output_dir: str):
        """Visualize the stats and save the plots to the output directory"""

        for stat in self.train_epoch_stats.keys():
            plt.clf()
            plt.plot(self.train_epoch_stats[stat])
            plt.plot(self.test_epoch_stats[stat])
            plt.title(stat)
            plt.savefig(f"{output_dir}/plots/{stat}.png")
