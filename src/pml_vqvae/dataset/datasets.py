from torch.utils.data import Dataset
import os
import torch
from torchvision.io import decode_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageNet


class ImageNetDataset(Dataset):
    """ImageNet Dataset

    Args:
        root_dir (str): Path to the ImageNet dataset
        train (bool): Whether to load the training or validation set
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir: str, train: bool = True, transform=None):
        super().__init__()
        if train:
            self.root_dir = os.path.join(root_dir, "train")
            self.imagenet = ImageNet(
                root_dir, "train"
            )  # remedy to get dataset information (i.e class names)
        else:
            self.root_dir = os.path.join(root_dir, "val")
            self.imagenet = ImageNet(
                root_dir, "val"
            )  # remedy to get dataset information (i.e class names)

        self.train = train
        self.transform = transform
        self.image_paths = []

    def __len__(self):
        return len(self.imagenet.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagenet.imgs[idx]

        # image_channels x image_height x image_width
        image = decode_image(img_path, mode="RGB")

        if self.transform:
            image = self.transform(image)

        return image, image

    def export_class_dist(self, outfile="./imagenet_dist"):
        fig = plt.figure(figsize=(10, 10))

        plt.hist(self.imagenet.targets, bins=self.imagenet.targets[-1] + 1)
        plt.title("Number of samples per class")
        plt.xlabel("Class")
        plt.ylabel("Number of samples")

        if outfile is not None:
            fig.savefig(outfile, bbox_inches="tight")
        else:
            plt.show()

    def info(self):
        info = {}

        # Number of samples
        info["n_samples"] = len(self.imagenet.imgs)

        # counts is ordered according to the labels 0...1000, thus we can infer the class in counts from its position
        classes, counts = np.unique(np.array(self.imagenet.targets), return_counts=True)

        # Number of classes
        info["n_classes"] = len(classes)

        # class with highest number of samples
        info["highest_sample_per_class"] = [{"n": np.max(counts)}]

        # class with lowest number of samples
        info["lowest_sample_per_class"] = [{"n": np.min(counts)}]

        # mean number of samples
        info["mean_samples_per_class"] = np.mean(counts)

        # range of image sizes

        pass


class CifarDataset(Dataset):
    """CIFAR-10 Dataset

    Args:
        train (bool): Whether to load the training or test set
        transform (callable, optional): Optional transform to be applied on a sample.
    """

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

    def export_class_dist(self):
        pass

    def info(self):
        pass


if __name__ == "__main__":
    dir = "/home/space/datasets/imagenet_torchvision/data"

    data = ImageNetDataset(dir)

    data.export_class_dist()

    print("hallo")
