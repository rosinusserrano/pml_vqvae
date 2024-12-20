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

    # required
    experiment_name: str
    dataset: str
    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float

    # experiment optionals
    description: str | None = None
    output_dir: str | None = None
    test_interval: int | None = None
    vis_train_interval: int | None = None
    wandb_log: bool = True

    # data optionals
    n_train: int | None = None
    n_test: int | None = None
    seed: int | None = None
    class_idx: int | None = None

    # model optionals (some model require a config, others don't)
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
            TrainConfig: Configuration object
        """

        instance = cls(**config)
        instance.make_directories()
        instance.integrity_check()

        return instance

    def make_directories(self):
        outdir = f"artifacts/{self.experiment_name}"

        if os.path.exists(outdir):  # if dir exists, create new with "_n" suffix
            artifacts_dir = os.listdir("artifacts")
            dirs_with_same_name = [
                d for d in artifacts_dir if self.experiment_name in d
            ]
            n = len(dirs_with_same_name)
            outdir = f"{outdir}_{n}"

        self.output_dir = outdir
        os.makedirs(f"{outdir}/models/", exist_ok=True)
        os.makedirs(f"{outdir}/visuals/", exist_ok=True)
        os.makedirs(f"{outdir}/plots/", exist_ok=True)

    def integrity_check(self):
        """Check if the configuration is complete"""
        if self.dataset not in AVAIL_DATASETS:
            raise ValueError(
                f"Dataset {self.dataset} is not available. Choose from {AVAIL_DATASETS}"
            )

        if self.model_name not in AVAIL_MODELS:
            raise ValueError(
                f"Model {self.model_name} is not available. Choose from {AVAIL_MODELS}"
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
