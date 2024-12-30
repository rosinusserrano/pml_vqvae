from torch.utils.data import Dataset
import torch
import numpy as np
from numpy.lib.npyio import NpzFile
import os


class LatentDatasetGenerator:
    def __init__(self, max_per_file: int = 1):
        self.latents = np.empty((0, 32, 32))
        self.labels = np.array([])
        self.max_per_file = max_per_file

    def add_latent(self, latents: np.array, labels: np.array):
        self.latents = np.append(self.latents, latents, axis=0)
        self.labels = np.append(self.labels, labels)
        print(f"Added {len(latents)} latents to dataset")

    def save(self, path: str, name: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Save dataset to file in chunks of not more than MAX_PER_FILE
        for i in range(0, len(self.latents), self.max_per_file):
            file_path = os.path.join(path, f"{name}_{(i // self.max_per_file):8d}.npz")
            np.savez(
                file_path,
                latents=self.latents[i : i + self.max_per_file],
                labels=self.labels[i : i + self.max_per_file],
            )

        print(f"Saved dataset to {path}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


class LatentDataset(Dataset):
    def __init__(self, rootdir: str):
        self.rootdir = rootdir

    def __len__(self):
        return len(os.listdir(self.rootdir))

    def __getitem__(self, index: int):
        with np.load(os.listdir(self.rootdir)[index]) as npzfile:
            latents = npzfile["latents"]
            labels = npzfile["labels"]

        return torch.tensor(latents), torch.tensor(labels)
