from torchvision.datasets import ImageNet, CIFAR10
from torchvision.transforms import ToTensor, RandomCrop
from torchvision.io import decode_image
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from visuals import show_image_grid

DATASET_DIR = "/home/space/datasets/"
DATASET_NAMES = ["imagenet", "imagenet_mini", "cifar"]


class ImageNetDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        super().__init__()
        if train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "val")
        self.train = train
        self.transform = transform
        self.image_paths = []

        for _, subdirs, _ in os.walk(self.root_dir):
            for dir in subdirs:
                for subdir, _, files in os.walk(os.path.join(self.root_dir, dir)):
                    for file in files:
                        if file.endswith((".JPEG")):
                            self.image_paths.append(os.path.join(subdir, file))

        print(f"paths: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        # image_channels x image_height x image_width
        image = decode_image(img_name)

        if self.transform:
            image = self.transform(image)

        return image, image


class CifarDataset(Dataset):
    def __init__(self, train: bool = True, transform=None):
        super().__init__()
        self.root_dir = "/home/space/datasets/cifar10/processed"
        if train:
            filename = os.path.join(self.root_dir, "training.npz")
        else:
            filename = os.path.join(self.root_dir, "test.npz")
        self.transform = transform
        self.images = np.load(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image, image


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
        train_set = CifarDataset(DATASET_DIR, split="train", transform=transformation)

        test_set = CifarDataset(DATASET_DIR, split="test", transform=transformation)

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

    return train_loader, test_loader, train_set, test_set


if __name__ == "__main__":

    t, te, a, b = load_data(
        "imagenet_mini",
        batch_size=20,
        transformation=RandomCrop(128, pad_if_needed=True),
    )

    # train_set = ImageNet(root=DATASET_DIR + "imagenet_torchvision/data", split="train")
    # loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=1, shuffle=True, num_workers=1
    # )

    sample_images = next(iter(t))[0]

    show_image_grid(sample_images, outfile="test.png")
