from torch.utils.data import Dataset
import os
import torch
from torchvision.io import decode_image
import numpy as np


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

        # walk through the subdirectories an save all image paths
        for _, subdirs, _ in os.walk(self.root_dir):
            for dir in subdirs:
                for subdir, _, files in os.walk(os.path.join(self.root_dir, dir)):
                    for file in files:
                        if file.endswith((".JPEG")):
                            self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]

        # image_channels x image_height x image_width
        image = decode_image(img_name, mode="RGB")

        if self.transform:
            image = self.transform(image)

        return image, image


class CifarDataset(Dataset):
    def __init__(self, train: bool = True, transform=None):
        super().__init__()
        self.root_dir = "/home/space/datasets/cifar10/processed"
        if train:
            self.filename = os.path.join(self.root_dir, "training.npz")
        else:
            self.filename = os.path.join(self.root_dir, "test.npz")
        self.transform = transform

        # open npz file
        self.images = np.load(self.filename)["data"]  # n x 32 x 32 x 3
        self.images = np.moveaxis(self.images, -1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torch.from_numpy(self.images[idx])

        if self.transform:
            image = self.transform(image)

        return image, image
