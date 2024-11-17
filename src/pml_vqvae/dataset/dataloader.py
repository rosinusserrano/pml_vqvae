from torchvision.transforms import RandomCrop
import torchvision
import torch

from pml_vqvae.visuals import show_image_grid
from torchvision.transforms import v2
from pml_vqvae.dataset.cifar10 import CifarDataset
from pml_vqvae.dataset.imagenet import ImageNetDataset

DATASET_NAMES = ["imagenet", "cifar"]


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
        if n_train is not None or n_test is not None:
            assert n_train % 1000 == 0, "n_train must be a multiple of 1000"
            assert n_test % 1000 == 0, "n_test must be a multiple of 1000"

            n_train = n_train // 1000
            n_test = n_test // 1000

        train_set = ImageNetDataset(
            split="train",
            samples_per_class=n_train,
            transform=transformation,
            seed=seed,
        )

        test_set = ImageNetDataset(
            split="val",
            samples_per_class=n_test,
            transform=transformation,
            seed=seed,
        )

    elif dataset == "cifar":
        # if n_train or n_test is specified, make sure it is a multiple of 10 as there are 10 classes
        if n_train is not None or n_test is not None:
            assert n_train % 10 == 0, "n_train must be a multiple of 10"
            assert n_test % 10 == 0, "n_test must be a multiple of 10"

            n_train = n_train // 10
            n_test = n_test // 10

        train_set = CifarDataset(
            split="train",
            samples_per_class=n_train,
            transform=transformation,
            seed=seed,
        )

        test_set = CifarDataset(
            split="test", samples_per_class=n_test, transform=transformation, seed=seed
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":

    transforms = v2.Compose(
        [
            # v2.RandomResizedCrop(size=(128, 128), antialias=True),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    t, te = load_data(
        "cifar",
        batch_size=15,
        seed=2020,
        transformation=transforms,
    )

    # train_set = ImageNet(root=DATASET_DIR + "imagenet_torchvision/data", split="train")
    # loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=1, shuffle=True, num_workers=1
    # )

    # set seed for reproducibility
    # torch.manual_seed(2809)

    sample_images = next(iter(t))[0]
    show_image_grid(sample_images, outfile=f"cifar_rbatch.png", rows=3, cols=5)
