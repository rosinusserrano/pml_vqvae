from torchvision.datasets import ImageNet, CIFAR10
from torchvision.transforms import ToTensor
import torch
import numpy as np

DATASET_DIR = "/home/space/datasets/"
DATASET_DIR = "./"
DATASET_NAMES = ["imagenet", "imagenet_mini", "cifar"]


def get_random_indices(high: int, n: int):

    rng = np.random.default_rng()

    return rng.choice(high, n, replace=False)


def load_data(
    dataset: str, transformation=None, n_train=None, n_test=None, num_workers=1
):
    train_set = None
    test_set = None

    if dataset not in DATASET_NAMES:
        print(
            f"The specified dataset is not supported. You can choose from {', '.join(DATASET_NAMES)}"
        )
        return None

    if transformation:
        transformation = transformation + [ToTensor()]
    else:
        transformation = [ToTensor()]

    if dataset == "imagenet":
        train_set = ImageNet(
            DATASET_DIR + "imagenet_torchvision/data", "train", transform=transformation
        )
        test_set = ImageNet(
            DATASET_DIR + "imagenet_torchvision/data", "val", transform=transformation
        )

    elif dataset == "imagenet_mini":
        train_set = ImageNet(
            DATASET_DIR + "imagenet_torchvision/imagenet100",
            "train",
            transform=transformation,
            download=True,
        )
        test_set = ImageNet(
            DATASET_DIR + "imagenet_torchvision/imagenet100",
            "val",
            transformat=transformation,
            download=True,
        )

    elif dataset == "cifar":
        train_set = CIFAR10(
            DATASET_DIR + "cifar10", train=True, transform=transformation
        )
        test_set = CIFAR10(
            DATASET_DIR + "cifar10", train=False, transform=transformation
        )

    if n_train != None:
        # sample random indices
        indices = get_random_indices(len(train_set), n_train)
        train_set = torch.utils.data.Subset(train_set, indices)

    if n_test != None:
        # sample random indices
        indices = get_random_indices(len(test_set), n_test)
        test_set = torch.utils.data.Subset(test_set, indices)

    # TODO: loader return labels as target for now but should return the image as well
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=True, num_workers=num_workers
    )

    return train_loader, test_loader


t, te = load_data("cifar")
