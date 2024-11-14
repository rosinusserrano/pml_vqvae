import matplotlib.pyplot as plt


class StatsKeeper:
    """Class to keep track of stats during training and testing

    Args:
        output_dir (str, optional): The output directory to save the plots to. Defaults to None.

    Raises:
        ValueError: If output_dir is not set
    """

    def __init__(self, output_dir: str = None):

        self.output_dir = output_dir

        self.train_epoch_stats = {}
        self.train_batch_stats = {}

        self.test_epoch_stats = {}
        self.test_batch_stats = {}

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
            train_batch_stats = {
                k: sum(v) / len(v) for k, v in self.train_batch_stats.items()
            }

            for key, value in train_batch_stats.items():
                self.train_epoch_stats.setdefault(key, []).append(value)

            self.train_batch_stats = {}
        else:
            test_batch_stats = {
                k: sum(v) / len(v) for k, v in self.test_batch_stats.items()
            }

            for key, value in test_batch_stats.items():
                self.test_epoch_stats.setdefault(key, []).append(value)

            self.test_batch_stats = {}

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
