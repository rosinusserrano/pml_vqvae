import json
from torch.utils.data import Dataset
import os
import torch
import numpy as np


def create_cifar_subset(
    file_dir: str, n_samples: int, seed: int = None, class_idx_list=None
):
    """Create a subset of the CIFAR-10 dataset with n_samples per class. The subset
    is created by randomly selecting n_samples images from each class. Thus the image
    distribution is uniform.

    Args:

        file_dir (str): Path to the CIFAR-10 dataset
        n_samples (int): Number of samples per class
        seed (int, optional): Seed for reproducibility. Defaults to None.
        class_idx_list (list, optional): List of class indices to load. Defaults to None.

    Returns:
        np.array: Subset of the CIFAR-10 dataset
    """

    data = np.load(file_dir)
    full_imgs = data["data"]  # n x 32 x 32 x 3
    full_imgs = np.moveaxis(full_imgs, -1, 1)

    full_labels = data["labels"]  # n

    sorted_idx = np.argsort(full_labels)

    sorted_labels = full_labels[sorted_idx]
    sorted_images = full_imgs[sorted_idx]

    img_subset = []
    img_pointer = 0
    labels = []
    for class_idx in range(10):
        class_imgs = []
        for idx in range(img_pointer, len(sorted_labels)):
            if sorted_labels[idx] != class_idx or idx == len(sorted_labels) - 1:
                img_pointer = idx

                # if only select specific classes and its at current iteration not that class, skip
                if class_idx_list and class_idx not in class_idx_list:
                    break

                if n_samples and len(class_imgs) > n_samples:
                    np.random.seed(seed=seed)
                    indices = np.random.choice(
                        len(class_imgs), n_samples, replace=False
                    )
                    img_subset.extend([class_imgs[i] for i in indices])
                    labels.extend([class_idx] * n_samples)
                else:
                    img_subset.extend(class_imgs)
                    labels.extend([class_idx] * len(class_imgs))
                break
            else:
                class_imgs.append(sorted_images[idx])

    img_subset = np.array(img_subset)
    labels = np.array(labels)

    return img_subset, labels


class CifarDataset(Dataset):
    """CIFAR-10 dataset

    Args:
        split (str, optional): Split of the dataset. Either 'train' or 'test'. Defaults to 'train'.
        samples_per_class (int, optional): Number of samples per class. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        transform ([type], optional): Transformation to apply to the data. Defaults to None.
        class_idx ([type], optional): List of class indices to load. Defaults to None.
    """

    def __init__(
        self,
        split: str = "train",
        samples_per_class: int = None,
        seed: int = None,
        transform=None,
        class_idx=None,
    ):
        super().__init__()
        self.root_dir = "/home/space/datasets/cifar10/processed"
        if split == "train":
            self.filename = os.path.join(self.root_dir, "training.npz")
        elif split == "test":
            self.filename = os.path.join(self.root_dir, "test.npz")
        else:
            raise ValueError("split must be either 'train' or 'test'")

        self.transform = transform

        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        if samples_per_class is not None or class_idx is not None:
            self.images, self.labels = create_cifar_subset(
                self.filename,
                samples_per_class,
                seed=seed,
                class_idx_list=class_idx,
            )
        else:
            # open npz file
            data = np.load(self.filename)
            self.images = data["data"]  # n x 32 x 32 x 3
            self.images = np.moveaxis(self.images, -1, 1)

            self.labels = data["labels"]  # n

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torch.from_numpy(self.images[idx]).to(torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, image

    def summary(self, outfile="./cifar_summary.json"):
        """Create a summary of the dataset

        Args:
            outfile (str, optional): Path to save the summary. Defaults to "./cifar_summary.json".

        Returns:
            dict: Summary of the dataset
        """
        info = {}

        # Number of samples
        info["n_samples"] = len(self.images)

        # Number of classes
        info["n_classes"] = len(self.classes)

        # mean number of samples
        # info["samples_per_class"] =

        # write to json file
        if outfile is not None:
            with open(outfile, "w") as f:
                json.dump(info, f)

        return info


if __name__ == "__main__":

    data = CifarDataset(split="train", samples_per_class=1)
