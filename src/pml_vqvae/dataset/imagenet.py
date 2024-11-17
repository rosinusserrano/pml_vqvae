import json
from torch.utils.data import Dataset
import os
import torch
from torchvision.io import decode_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageNet
import imagesize
import torchvision.transforms as v2

DATASET_DIR = "/home/space/datasets/"


def create_imagenet_subset(root_dir: str, n_samples: int, split: str, seed: int = None):
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

                # choose n_samples random images from the subset
                if len(class_imgs) > n_samples:
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
    ):
        super().__init__()

        # root directory of the dataset
        self.root_dir = os.path.join(root_dir, split)

        if samples_per_class is not None:
            self.imagenet = create_imagenet_subset(
                root_dir, samples_per_class, split, seed=seed
            )
        else:
            self.imagenet = ImageNet(root_dir, split)

        self.split = split
        self.transform = transform
        self.image_paths = []

        if samples_per_class is not None:
            self.samples_per_class = samples_per_class

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

    def image_size_from_file(self):
        with open("sizes.json") as json_file:
            sizes = json.load(json_file)

        class_sizes = []

        for class_idx, sizes in sizes.items():
            sizes = np.array(sizes)
            mean_size = np.mean(sizes, axis=0)
            max_size = np.max(sizes, axis=0)
            min_size = np.min(sizes, axis=0)

            class_sizes.append((class_idx, [*mean_size, *max_size, *min_size]))

        # sort by class index
        class_sizes = sorted(class_sizes, key=lambda x: x[0])

        # as a numpy array of shape (n_classes, 9)
        class_sizes = np.array([x[1] for x in class_sizes])

        # save to file
        np.save("imagenet_sizes.npy", class_sizes)

    # get class with the largest max image width
    def max_class(self):
        sizes = np.load("imagenet_sizes.npy")

        max_width_class = sizes[:, 2].argmax()
        print(f"Class with largest max image width: {max_width_class}")

        max_height_class = sizes[:, 3].argmax()
        print(f"Class with largest max image height: {max_height_class}")

        min_width_class = sizes[:, 4].argmin()
        print(f"Class with smallest min image width: {min_width_class}")

        min_height_class = sizes[:, 5].argmin()
        print(f"Class with smallest min image height: {min_height_class}")

        # find and plot the max images
        max_width_images = []
        max_height_images = []
        min_width_images = []
        min_height_images = []

        for img_path, class_idx in self.imagenet.imgs:
            if class_idx == max_width_class:
                max_width_images.append(img_path)
            elif class_idx == max_height_class:
                max_height_images.append(img_path)
            elif class_idx == min_width_class:
                min_width_images.append(img_path)
            elif class_idx == min_height_class:
                min_height_images.append(img_path)

        l_width = 0
        l_width_img = ""
        l_height_img = ""
        s_width_img = ""
        s_height_img = ""
        l_height = 0
        for img in max_width_images:
            width, height = imagesize.get(img)
            if width > l_width:
                l_width = width
                l_width_img = img

        for img in max_height_images:
            width, height = imagesize.get(img)
            if height > l_height:
                l_height = height
                l_height_img = img

        for img in min_width_images:
            width, height = imagesize.get(img)
            if width < l_width:
                l_width = width
                s_width_img = img

        for img in min_height_images:
            width, height = imagesize.get(img)
            if height < l_height:
                l_height = height
                s_height_img = img

        print(f"Max width: {l_width}")
        print(l_width_img)
        plt.imshow(decode_image(l_width_img, mode="RGB").permute(1, 2, 0).numpy())
        plt.title("/".join(l_height_img.split("/")[-2:]))
        plt.savefig("resources/max_width_img.png")

        print(f"Max height: {l_height}")
        print(l_height_img)
        plt.imshow(decode_image(l_height_img, mode="RGB").permute(1, 2, 0).numpy())
        plt.title("/".join(l_height_img.split("/")[-2:]))
        plt.savefig("resources/max_height_img.png")

        print(f"Min width: {l_width}")
        print(s_width_img)
        plt.imshow(decode_image(s_width_img, mode="RGB").permute(1, 2, 0).numpy())
        plt.title("/".join(s_width_img.split("/")[-2:]))
        plt.savefig("resources/min_width_img.png")

        print(f"Min height: {l_height}")
        print(s_height_img)
        plt.imshow(decode_image(s_height_img, mode="RGB").permute(1, 2, 0).numpy())
        plt.title("/".join(s_height_img.split("/")[-2:]))
        plt.savefig("resources/min_height_img.png")

    def image_close_to_32x32(self):
        for img, class_idx in self.imagenet.imgs:
            width, height = imagesize.get(img)

            if class_idx % 50 == 0:
                print(f"Class {class_idx}")

            if (width >= 30 and width <= 34) or (height >= 30 and height <= 34):
                print(img)
                print(width, height)

                # plot original image and random crop resized image
                image = decode_image(img, mode="RGB")
                image_resized = v2.RandomResizedCrop(128)(image)

                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(image.permute(1, 2, 0).numpy())
                ax[0].set_title("Original image")

                ax[1].imshow(image_resized.permute(1, 2, 0).numpy())
                ax[1].set_title("Random Resized Cropped image")

                fig.suptitle("/".join(img.split("/")[-2:]))
                plt.tight_layout()
                plt.savefig("resources/close_to_32x32.png")
                break

    def randomcropresize(self):
        img_small = "/home/space/datasets/imagenet_torchvision/data/train/n02783161/n02783161_4703.JPEG"
        image_large = "/home/space/datasets/imagenet_torchvision/data/train/n03933933/n03933933_10972.JPEG"

        image = decode_image(img_small, mode="RGB")
        image_resized = v2.RandomResizedCrop(128)(image)

        # plot original and resized image side by side
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image.permute(1, 2, 0).numpy())
        ax[0].set_title("Original image")

        ax[1].imshow(image_resized.permute(1, 2, 0).numpy())
        ax[1].set_title("Random Resized Cropped image")

        fig.suptitle("/".join(img_small.split("/")[-2:]))
        plt.tight_layout()
        plt.savefig("resources/random_resized_crop_small.png")

        image = decode_image(image_large, mode="RGB")
        image_resized = v2.RandomResizedCrop(128)(image)

        # clear figure
        plt.clf()

        # plot original and resized image side by side
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image.permute(1, 2, 0).numpy())
        ax[0].set_title("Original image")

        ax[1].imshow(image_resized.permute(1, 2, 0).numpy())
        ax[1].set_title("Random Resized Cropped image")

        fig.suptitle("/".join(image_large.split("/")[-2:]))
        plt.tight_layout()
        plt.savefig("resources/random_resized_crop_large.png")

    def create_size_errorbar(self):
        sizes = np.load("imagenet_sizes.npy")

        mean = sizes.mean(axis=0)
        print(sizes.mean(axis=0)[:2])
        print(sizes.max(axis=0)[2:4])
        print(sizes.min(axis=0)[4:])

        fig, ax = plt.subplots(2)

        # plot mean image width per class with min max deviation
        ax[0].errorbar(
            np.arange(sizes.shape[0]),
            sizes[:, 0],
            yerr=[sizes[:, 0] - sizes[:, 4], sizes[:, 2] - sizes[:, 0]],
            fmt="o",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax[0].plot(
            np.arange(sizes.shape[0]),
            [mean[0]] * sizes.shape[0],
            "r--",
            label="Mean= {:.2f}".format(mean[0]),
        )
        ax[0].plot(
            np.arange(sizes.shape[0]),
            [mean[2]] * sizes.shape[0],
            "g--",
            label="Max= {:.2f}".format(mean[2]),
        )
        ax[0].plot(
            np.arange(sizes.shape[0]),
            [mean[4]] * sizes.shape[0],
            "b--",
            label="Min= {:.2f}".format(mean[4]),
        )
        ax[0].set_title("Mean image width per class")
        ax[0].set_xlabel("Class ID")
        ax[0].set_ylabel("Image width in px")
        ax[0].legend()

        plt.clf()

        # plot mean image height per class with min max deviation
        ax[1].errorbar(
            np.arange(sizes.shape[0]),
            sizes[:, 1],
            yerr=[sizes[:, 1] - sizes[:, 5], sizes[:, 3] - sizes[:, 1]],
            fmt="o",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax[1].plot(
            np.arange(sizes.shape[0]),
            [mean[1]] * sizes.shape[0],
            "r--",
            label="Mean= {:.2f}".format(mean[1]),
        )
        ax[1].plot(
            np.arange(sizes.shape[0]),
            [mean[3]] * sizes.shape[0],
            "g--",
            label="Max= {:.2f}".format(mean[3]),
        )
        ax[1].plot(
            np.arange(sizes.shape[0]),
            [mean[5]] * sizes.shape[0],
            "b--",
            label="Min= {:.2f}".format(mean[5]),
        )
        ax[1].set_title("Mean image height per class")
        ax[1].set_xlabel("Class ID")
        ax[1].set_ylabel("Image height in px")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("resources/imagenet_sizes_errorbar.png")

        # clear figure
        plt.clf()

        # create histogram plot of image mean image width and height
        fig, ax = plt.subplots(2)

        ax[0].hist(sizes[:, 0], bins=50)
        ax[0].set_title("Mean image width distribution")
        ax[0].set_xlabel("Image width")
        ax[0].set_ylabel("Number of classes")
        # vertical line for mean
        ax[0].axvline(
            mean[0], color="r", linestyle="--", label="Mean= {:.2f}".format(mean[0])
        )
        ax[0].legend()

        ax[1].hist(sizes[:, 1], bins=50)
        ax[1].set_title("Mean image height distribution")
        ax[1].set_xlabel("Image height")
        ax[1].set_ylabel("Number of classes")
        # vertical line for mean
        ax[1].axvline(
            mean[1], color="r", linestyle="--", label="Mean= {:.2f}".format(mean[1])
        )
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("resources/imagenet_sizes_histogram.png")

    def image_size_summary(self):
        """Compute the image size distribution of the dataset.

        Returns:
            dict: Image size distribution
        """
        print("Start computing image size distribution...")

        class_sizes = []
        sizes = []
        curr_idx = 0
        for img_path, class_idx in self.imagenet.imgs:
            if class_idx <= 750:
                curr_idx = 751
                continue

            if class_idx != curr_idx:
                sizes_np = np.array(sizes)
                mean_size = np.mean(sizes_np, axis=0)
                max_size = np.max(sizes_np, axis=0)
                min_size = np.min(sizes_np, axis=0)

                class_sizes.append([*mean_size, *max_size, *min_size])

                print(f"Class {curr_idx} done")

                sizes = []
                curr_idx += 1

            # returns (width, height)
            size = imagesize.get(img_path)

            sizes.append(size)

        sizes_np = np.array(sizes)
        mean_size = np.mean(sizes_np, axis=0)
        max_size = np.max(sizes_np, axis=0)
        min_size = np.min(sizes_np, axis=0)

        class_sizes.append([*mean_size, *max_size, *min_size])

        sizes = np.array(class_sizes)

        # save to file
        np.save("resources/imagenet_sizes_4.npy", sizes)

        return

    def export_class_dist(self, outfile="./imagenet_dist"):
        """Export the class distribution of the dataset as a histogram.

        Args:
            outfile (str, optional): Path to save the histogram. Defaults to "./imagenet_dist".
        """

        fig = plt.figure()

        plt.hist(self.imagenet.targets, bins=self.imagenet.targets[-1] + 1)
        plt.title("Number of samples per class")
        plt.xlabel("Class ID")
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

    data = ImageNetDataset(split="train")
    data.image_close_to_32x32()
    data.randomcropresize()
