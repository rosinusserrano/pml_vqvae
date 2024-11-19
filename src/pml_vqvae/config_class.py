from pml_vqvae.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.baseline.vae import BaselineVariationalAutoencoder
import yaml
import os

AVAIL_DATASETS = ["cifar", "imagenet"]
AVAIL_MODELS = ["vae", "autoencoder"]


class Config:
    """Configuration class for the training process"""

    def __init__(self):
        """Initialize the a default configuration class"""

        # experiment
        self.name = None
        self.description = None
        self.output_dir = None
        self.test_interval = None
        self.vis_train_interval = None
        self.wandb_log = None

        # data
        self.dataset = None
        self.n_train = None
        self.n_test = None
        self.seed = None

        # train
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

        # model
        self.model_name = None

    def to_dict(self):
        """Convert the configuration to a dictionary

        Returns:
            dict: Configuration dictionary
        """
        return self.__dict__

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

                if key == "name":
                    outdir = f"artifacts/{value}_output"
                    setattr(conf, "output_dir", outdir)

                    os.makedirs(f"{outdir}/models/", exist_ok=True)
                    os.makedirs(f"{outdir}/visuals/", exist_ok=True)
                    os.makedirs(f"{outdir}/plots/", exist_ok=True)

        conf.integrity_check()

        return conf

    def integrity_check(self):
        """Check if the configuration is complete"""
        for key, value in self.__dict__.items():
            if value is None:
                if key not in [
                    "description",
                    "test_interval",
                    "vis_train_interval",
                    "n_train",
                    "n_test",
                ]:
                    raise ValueError(f"Parameter {key} is not set in the configuration")

            if key == "dataset" and value not in AVAIL_DATASETS:
                raise ValueError(
                    f"Dataset {value} is not available. Choose from {AVAIL_DATASETS}"
                )

            if key == "model_name" and value not in AVAIL_MODELS:
                raise ValueError(
                    f"Model {value} is not available. Choose from {AVAIL_MODELS}"
                )

    def __str__(self):
        """Return a string representation of the configuration"""
        return str(self.__dict__)

    def get_model(self):
        """Return the model based on the model name"""
        if self.model_name == "vae":
            return BaselineVariationalAutoencoder()
        elif self.model_name == "autoencoder":
            return BaselineAutoencoder()

        raise ValueError(f"Model {self.model_name} is not available.")


if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config = Config.from_dict(config)

    print("done")
