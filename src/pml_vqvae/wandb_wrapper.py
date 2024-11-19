import wandb
import os
from pml_vqvae.config_class import Config
import numpy as np


class WANDBWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.log = config.wandb_log

        self.train_expl = None
        self.test_expl = None

        self.train_example_table = None
        self.test_example_table = None

    def init(self, model):
        if self.log:
            wandb.login(key=os.environ["WANDB_API_KEY"])  # your api key
            wandb.init(
                project="pml_vqvae", name=self.config.name, config=self.config.to_dict()
            )  # DON'T change the project name
            wandb.watch(model, log_freq=1)  # logs the gradients, too

            self.train_example_table = wandb.Table(
                columns=["Epoch", "Input", "Reconstructions"]
            )
            self.test_example_table = wandb.Table(
                columns=["Epoch", "Input", "Reconstructions"]
            )

    def save_model(self, path: str):
        if self.log:
            wandb.save(path)

    def prepare_stats(self, train_stats, test_stats, example_cnt):
        train_stats = {"train/" + k: v for k, v in train_stats.items()}
        test_stats = {"test/" + k: v for k, v in test_stats.items()}
        return train_stats | test_stats | {"example_cnt": example_cnt}

    def log_epoch(self, stats, epoch, log_vis):

        if self.log:
            epoch_stats = self.prepare_stats(*stats)

            wandb.log(epoch_stats, step=epoch)

            if log_vis:
                self.test_example_table.add_data(
                    epoch, self.test_expl["Input"], self.test_expl["Reconstructions"]
                )
                self.train_example_table.add_data(
                    epoch, self.train_expl["Input"], self.train_expl["Reconstructions"]
                )

    def finish(self):
        """Finish the logging process"""

        if self.log:
            if self.train_example_table:
                wandb.log({"train_examples": self.train_example_table})
            if self.test_example_table:
                wandb.log({"test_examples": self.test_example_table})
            wandb.finish()

    def construct_examples(self, batch, output, train=True, max_examples=5):

        if isinstance(output, tuple):
            output = output[0]

        num_examples = min(len(batch), max_examples)

        payload = {
            "Input": [
                wandb.Image(np.moveaxis(batch[i].cpu().detach().numpy(), 0, -1))
                for i in range(num_examples)
            ],
            "Reconstructions": [
                wandb.Image(np.moveaxis(output[i].cpu().detach().numpy(), 0, -1))
                for i in range(num_examples)
            ],
        }

        if train:
            self.train_expl = payload
        else:
            self.test_expl = payload
