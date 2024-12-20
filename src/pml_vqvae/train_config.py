import os
from dataclasses import dataclass, asdict

from pml_vqvae.models.baseline.autoencoder import BaselineAutoencoder
from pml_vqvae.models.baseline.vae import BaselineVariationalAutoencoder
from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig

AVAIL_DATASETS = ["cifar", "imagenet"]
AVAIL_MODELS = ["vae", "autoencoder", "vqvae"]


@dataclass
class TrainConfig:
    """Configuration class for the training process"""

    # experiment
    experiment_name: str
    description: str | None = None
    output_dir: str | None = None
    test_interval: int | None = None
    vis_train_interval: int | None = None
    wandb_log: bool = True

    # data
    dataset: str
    n_train: int | None = None
    n_test: int | None = None
    seed: int | None = None
    class_idx: int | None = None

    # train
    batch_size: int
    epochs: int
    learning_rate: float

    # model
    model_name: str
    model_config: dict | None = None

    def to_dict(self):
        """Convert the configuration to a dictionary

        Returns:
            dict: Configuration dictionary
        """
        return asdict(self)

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
                    "class_idx",
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

        if self.model_name == "autoencoder":
            return BaselineAutoencoder()

        if self.model_name == "vqvae":
            if self.model_config is None:
                raise ValueError("VQ-VAE needs model config!")
            config = VQVAEConfig(**self.model_config)
            return VQVAE(config)

        if self.model_name == "pixelcnn":
            if self.model_config is None:
                raise ValueError("PixelCNN needs model config!")
            config = PixelCNNConfig(**self.model_config)
            return PixelCNN(config)

        raise ValueError(f"Model {self.model_name} is not available.")
