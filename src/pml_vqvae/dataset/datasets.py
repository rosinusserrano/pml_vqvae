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

    def export_pxl_size_per_class(self, outfile="./imagenet_img_sizes"):

        pixel_counts = []
        sizes = []

        # iterate over all images classe and get their size
        for wnid in self.imagenet.wnids:
            # get class id
            class_id = self.imagenet.wnid_to_idx[wnid]

            # get all images paths
            img_paths = [img[0] for img in self.imagenet.imgs if img[1] == class_id]

            class_sizes = []
            class_pixel_count = []
            # get image sizes
            for image_path in img_paths:
                img = decode_image(image_path, mode="RGB")
                img_size = img.size[-2]
                pixel_count = img_size[0] * img_size[1]

                class_sizes.append(img_size)
                class_pixel_count.append(pixel_count)

            sizes.append(class_sizes)
            pixel_counts.append(class_pixel_count)

        pixels_per_class = [
            [np.mean(counts), np.max(counts), np.min(counts)] for counts in pixel_counts
        ]
        sizes_per_class = [
            [
                [np.mean(width), np.max(width), np.min(width)],
                [np.mean(height), np.max(height), np.min(height)],
            ]
            for width, height in sizes
        ]

        # plot an errorbar plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for i, (mean, max, min) in enumerate(pixels_per_class):
            ax.errorbar(
                i,
                mean,
                yerr=[[mean - min], [max - mean]],
                fmt="o",
            )

        ax.set_xticks(range(len(self.imagenet.classes)))
        ax.set_xticklabels(self.imagenet.classes, rotation=90)
        ax.set_ylabel("Number of pixels")
        ax.set_title("Number of pixels per class")

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

        # 5 classes with highest number of samples
        info["highest_sample_per_class"] = {
            "class": classes[np.argsort(counts)[-5:]],
            "count": counts[np.argsort(counts)[-5:]],
            "class_name": [
                self.imagenet.classes[i] for i in classes[np.argsort(counts)[-5:]]
            ],
        }

        # 5 classes with lowest number of samples
        info["lowest_sample_per_class"] = {
            "class": classes[np.argsort(counts)[:5]],
            "count": counts[np.argsort(counts)[:5]],
            "class_name": [
                self.imagenet.classes[i] for i in classes[np.argsort(counts)[:5]]
            ],
        }

        # mean number of samples
        info["mean_samples_per_class"] = np.mean(counts)

        # range of image sizes
        sizes = np.array([img[0].size for img in self.imagenet.imgs])

        # mean image size for every class

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
