from torchvision.transforms import RandomCrop, CenterCrop
import torch

import numpy as np
from visuals import show_image_grid

from datasets import ImageNetDataset, CifarDataset

DATASET_DIR = "/home/space/datasets/"
DATASET_NAMES = ["imagenet", "imagenet_mini", "cifar"]


def get_random_indices(high: int, n: int, seed=None):

    if seed:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    return rng.choice(high, n, replace=False)


def load_data(
    dataset: str,
    transformation=None,
    n_train=None,
    n_test=None,
    num_workers=1,
    batch_size=32,
    shuffle=True,
    seed=None,
):
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
