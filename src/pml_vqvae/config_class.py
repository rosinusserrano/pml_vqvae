from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
import yaml

AVAIL_DATASETS = ["cifar", "imagenet"]
AVAIL_MODELS = ["vae", "autoencoder"]


MODEL = BaselineAutoencoder()
DATASET = "cifar"

EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9
N_TRAIN = 1000
N_TEST = 1000
BATCH_SIZE = 32

SEED = 2024


class Config:
    """Configuration class for the training process"""

    def __init__(self):
        """Initialize the a default configuration class"""

        # experiment
        self.name = "default"
        self.description = "Default confirguration"
        self.save_dir = "checkpoints"
        self.log_dir = "logs"

        # data
        self.dataset = DATASET
        self.n_train = N_TRAIN
        self.n_test = N_TEST
        self.seed = SEED

        # train
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE

        # model
        self.model = MODEL

    @classmethod
    def from_dict(cls, config: dict):
        """Create a configuration object from a dictionary

        Args:
            config (dict): Configuration dictionary

        Returns:
            Config: Configuration object
        """

        conf = cls()
        for _, params in config.items():
            for key, value in params.items():
                if not hasattr(conf, key):
                    print(
                        f"Warning: {key} is not a valid parameter as part of the configuration. It will be ignored."
                    )
                setattr(conf, key, value)

                if key == "model_name":
                    if value == "autoencoder":
                        conf.model = BaselineAutoencoder()
                    elif value == "vae":
                        conf.model = BaselineVariationalAutoencoder()
                    else:
                        assert value in AVAIL_MODELS, "Unknown model"

                elif key == "dataset":
                    assert value in AVAIL_DATASETS, "Unknown dataset"

        return conf

    def summary(self):
        """Print a summary of the configuration"""
        print("Configuration:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")


if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config = Config.from_dict(config)

    print("done")
