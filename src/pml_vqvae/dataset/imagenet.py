import json
from torch.utils.data import Dataset
import os
import torch
from torchvision.io import decode_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageNet

DATASET_DIR = "/home/space/datasets/"


def create_imagenet_subset(
    root_dir: str,
    n_samples: int,
    split: str,
    seed: int = None,
    class_idx_list: list = None,
):
    """Create a subset of the ImageNet dataset with n_samples per class.

    Args:
        root_dir (str): Path to the ImageNet dataset
        n_samples (int): Number of samples per class
        split (str): Whether to load the training or test set
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        ImageNet: Subset of the ImageNet dataset
    """

    full_imagenet = ImageNet(root_dir, split)

    img_subset = []

    img_pointer = 0
    # iterate over all classes
    for class_idx in range(len(full_imagenet.classes)):
        class_imgs = []
        # iterate over all images starting from last pointer
        for idx in range(img_pointer, len(full_imagenet.imgs)):
            path, img_class_idx = full_imagenet.imgs[idx]

            # if not, then subset for the class is complete
            if img_class_idx != class_idx or idx == len(full_imagenet.imgs) - 1:
                img_pointer = idx  # save the pointer for the next class

                # if only select specific classes and its at current iteration not that class, skip
                if class_idx_list and class_idx not in class_idx_list:
                    break

                # choose n_samples random images from the subset
                if n_samples and len(class_imgs) > n_samples:
                    np.random.seed(seed)
                    indices = np.random.choice(
                        len(class_imgs), n_samples, replace=False
                    )
                    img_subset.extend([class_imgs[i] for i in indices])
                else:
                    img_subset.extend(class_imgs)

                break

            # if current image belongs to the current class, add it to the subset
            else:
                class_imgs.append((path, img_class_idx))

    subset_imagenet = full_imagenet
    subset_imagenet.imgs = img_subset
    subset_imagenet.samples = img_subset
    subset_imagenet.targets = [img[1] for img in img_subset]

    return subset_imagenet


class ImageNetDataset(Dataset):
    """ImageNet Dataset

    Args:
        root_dir (str): Path to the ImageNet dataset
        split (str, optional): Whether to load the training or test set. Defaults to True.
        samples_per_class (int, optional): Number of samples per class to load. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(
        self,
        root_dir: str = DATASET_DIR + "imagenet_torchvision/data",
        split: str = True,
        samples_per_class: int = None,
        seed: int = None,
        transform=None,
        class_idx: list = None,
    ):
        super().__init__()

        # root directory of the dataset
        self.root_dir = os.path.join(root_dir, split)

        if samples_per_class is not None or class_idx is not None:
            self.imagenet = create_imagenet_subset(
                root_dir, samples_per_class, split, seed=seed, class_idx=class_idx
            )
            self.samples_per_class = samples_per_class
        else:
            self.imagenet = ImageNet(root_dir, split)

        self.split = split
        self.transform = transform
        self.image_paths = []
        self.class_idx = class_idx

    def __len__(self):
        return len(self.imagenet.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagenet.imgs[idx][0]
        # image_channels x image_height x image_width
        image = decode_image(
            img_path,
            mode="RGB",
        ).to(torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, image

    def export_class_dist(self, outfile="./imagenet_dist"):
        """Export the class distribution of the dataset as a histogram.

        Args:
            outfile (str, optional): Path to save the histogram. Defaults to "./imagenet_dist".
        """

        fig = plt.figure(figsize=(10, 10))

        plt.hist(self.imagenet.targets, bins=self.imagenet.targets[-1] + 1)
        plt.title("Number of samples per class")
        plt.xlabel("Class")
        plt.ylabel("Number of samples")

        if outfile is not None:
            fig.savefig(outfile, bbox_inches="tight")
        else:
            plt.show()

    def summary(self, outfile="./imagenet_summary.json"):
        """Compute and print information about the dataset.

        Args:
            outfile (str, optional): Path to save the summary. Defaults to "./imagenet_summary.json".

        Returns:
            dict: Information about the dataset
        """

        print("Start computing dataset information...")

        info = {}

        # Number of samples
        info["n_samples"] = len(self.imagenet.imgs)

        # counts is ordered according to the labels 0...1000, thus we can infer the class in counts from its position
        classes, counts = np.unique(np.array(self.imagenet.targets), return_counts=True)

        # Number of classes
        info["n_classes"] = len(classes)

        # 5 classes with highest number of samples
        info["highest_sample_per_class"] = {
            "class": classes[np.argsort(counts)[-5:]].tolist(),
            "count": counts[np.argsort(counts)[-5:]].tolist(),
            "class_name": [
                self.imagenet.classes[i] for i in classes[np.argsort(counts)[-5:]]
            ],
        }

        # 5 classes with lowest number of samples
        info["lowest_sample_per_class"] = {
            "class": classes[np.argsort(counts)[:5]].tolist(),
            "count": counts[np.argsort(counts)[:5]].tolist(),
            "class_name": [
                self.imagenet.classes[i] for i in classes[np.argsort(counts)[:5]]
            ],
        }

        # mean number of samples
        info["mean_samples_per_class"] = np.mean(counts).item()

        # write to json file
        if outfile is not None:
            with open(outfile, "w") as f:
                json.dump(info, f)

        return info


if __name__ == "__main__":

    data = ImageNetDataset(split="train", samples_per_class=10)

    # info = data.summary()
