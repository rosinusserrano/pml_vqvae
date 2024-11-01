from torchvision.transforms import RandomCrop, CenterCrop
import torchvision
import torch

import numpy as np
from visuals import show_image_grid

from datasets import ImageNetDataset, CifarDataset

DATASET_DIR = "/home/space/datasets/"
DATASET_NAMES = ["imagenet", "imagenet_mini", "cifar"]


def get_random_indices(high: int, n: int, seed=None):
    """Generates n random indices from 0 to high-1 without replacement

    Args:
        high (int): highest index
        n (int): number of indices to generate
        seed (int, optional): Seed for reproducibilty . Defaults to None.

    Returns:
        list: List of random indices
    """

    if seed:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    return rng.choice(high, n, replace=False)


def load_data(
    dataset: str,
    transformation: torchvision.transforms = None,
    n_train: int = None,
    n_test: int = None,
    num_workers: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
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
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: Training data loader
        torch.utils.data.DataLoader: Test data loader
    """

    train_set = None
    test_set = None

    # if unknown dataset name
    if dataset not in DATASET_NAMES:
        print(
            f"The specified dataset is not supported. You can choose from {', '.join(DATASET_NAMES)}"
        )
        return None

    if dataset == "imagenet":
        train_set = ImageNetDataset(
            DATASET_DIR + "imagenet_torchvision/data", transform=transformation
        )

        test_set = ImageNetDataset(
            DATASET_DIR + "imagenet_torchvision/data",
            train=False,
            transform=transformation,
        )

    elif dataset == "imagenet_mini":
        train_set = ImageNetDataset(
            DATASET_DIR + "imagenet_mini", transform=transformation
        )
        test_set = ImageNetDataset(
            DATASET_DIR + "imagenet_mini", train=False, transform=transformation
        )

    elif dataset == "cifar":
        train_set = CifarDataset(transform=transformation)

        test_set = CifarDataset(train=False, transform=transformation)

    if n_train != None:
        # sample random indices
        indices = get_random_indices(len(train_set), n_train, seed=seed)
        train_set = torch.utils.data.Subset(train_set, indices)

    if n_test != None:
        # sample random indices
        indices = get_random_indices(len(test_set), n_test, seed=seed)
        test_set = torch.utils.data.Subset(test_set, indices)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":

    t, te = load_data(
        "cifar",
        batch_size=20,
        n_train=40,
        seed=2020,
        transformation=RandomCrop(128, pad_if_needed=True),
    )

    # train_set = ImageNet(root=DATASET_DIR + "imagenet_torchvision/data", split="train")
    # loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=1, shuffle=True, num_workers=1
    # )

    # set seed for reproducibility
    # torch.manual_seed(2809)
    sample_images = next(iter(t))[0]

    show_image_grid(sample_images, outfile="test.png")
