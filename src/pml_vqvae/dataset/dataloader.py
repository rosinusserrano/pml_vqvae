import torchvision
import torch

from torchvision.datasets import MNIST
from torchvision.transforms import v2

from pml_vqvae.dataset.cifar10 import CifarDataset
from pml_vqvae.dataset.imagenet import ImageNetDataset

DATASET_NAMES = ["imagenet", "cifar", "mnist"]


def load_data(
    dataset: str,
    transformation: torchvision.transforms = None,
    n_train: int = None,
    n_test: int = None,
    num_workers: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    class_idx: list = None,
    seed: int = None,
):
    """Load data from specified dataset

    Args:
        dataset (str): Name of the dataset
        transformation (torchvision.transforms, optional): Transformation to apply to the data. Defaults to None.
        n_train (int, optional): Number of training samples to load. Defaults to None.
        n_test (int, optional): Number of test samples to load. Defaults to None.
        num_workers (int, optional): Number of workers to use for data loading. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Shuffle data. Defaults to True.
        class_idx (list, optional): List of class indices to load. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: Training data loader
        torch.utils.data.DataLoader: Test data loader
    """

    train_set = None
    test_set = None

    if seed:
        torch.manual_seed(seed)

    # if unknown dataset name
    if dataset not in DATASET_NAMES:
        print(
            f"The specified dataset is not supported. You can choose from {', '.join(DATASET_NAMES)}"
        )
        return None

    if dataset == "imagenet":
        # if n_train or n_test is specified, make sure it is a multiple of 1000 as there are 1000 classes
        if n_train is not None:
            assert n_train % 1000 == 0, "n_train must be a multiple of 1000"
            n_train = n_train // 1000
        if n_test is not None:
            assert n_test % 1000 == 0, "n_test must be a multiple of 1000"
            n_test = n_test // 1000

        train_set = ImageNetDataset(
            split="train",
            samples_per_class=n_train,
            transform=transformation,
            seed=seed,
            class_idx=class_idx,
        )

        test_set = ImageNetDataset(
            split="val",
            samples_per_class=n_test,
            transform=transformation,
            seed=seed,
            class_idx=class_idx,
        )

    elif dataset == "cifar":
        # if n_train or n_test is specified, make sure it is a multiple of 10 as there are 10 classes
        if n_train is not None:
            assert n_train % 10 == 0, "n_train must be a multiple of 10"
            n_train = n_train // 10
        if n_test is not None:
            assert n_test % 10 == 0, "n_test must be a multiple of 10"
            n_test = n_test // 10

        train_set = CifarDataset(
            split="train",
            samples_per_class=n_train,
            transform=transformation,
            seed=seed,
            class_idx=class_idx,
        )

        test_set = CifarDataset(
            split="test",
            samples_per_class=n_test,
            transform=transformation,
            seed=seed,
            class_idx=class_idx,
        )

    elif dataset == "mnist":
        train_set = MNIST(
            root="/home/space/datasets",
            train=True,
            transform=v2.ToTensor(),
        )

        test_set = MNIST(
            root="/home/space/datasets",
            train=False,
            transform=v2.ToTensor(),
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader
