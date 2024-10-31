from torchvision.datasets import ImageNet, CIFAR10
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

DATASET_DIR = "/home/space/datasets/"
DATASET_NAMES = ["imagenet", "imagenet_mini", "cifar"]


class ImageNetDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super.__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith((".png", ".jpg", ".jpeg", ".JPEG"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image


class CifarDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super.__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = np.load(os.path.join(root_dir, split))

    def __len__(self):
        return len(self.images)

    def __getitem__(self):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        # image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image


def get_random_indices(high: int, n: int, seed):

    if seed:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng(seed=seed)

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

    # always add ToTensor transform
    # if transformation:
    #     transformation = transformation + [ToTensor()]
    # else:
    #     transformation = [ToTensor()]

    if dataset == "imagenet":
        train_set = ImageNetDataset(
            DATASET_DIR + "imagenet_torchvision/data/train", transform=transformation
        )

        test_set = ImageNetDataset(
            DATASET_DIR + "imagenet_torchvision/data/test", transform=transformation
        )

    elif dataset == "imagenet_mini":
        train_set = ImageNetDataset(
            DATASET_DIR + "imagenet_mini/train", transform=transformation
        )
        train_set = ImageNetDataset(
            DATASET_DIR + "imagenet_mini/val", transform=transformation
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

    return train_loader, test_loader


t, te = load_data("cifar")

print("trainset size: ", len(t.dataset))
print("testset size: ", len(te.dataset))
