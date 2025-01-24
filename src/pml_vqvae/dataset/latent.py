from torch.utils.data import Dataset
import torch
import numpy as np
from numpy.lib.npyio import NpzFile
import os


class LatentDatasetGenerator:
    def __init__(self, max_per_file: int, save_path: str):
        self.latents = np.empty((0, 32, 32))
        self.labels = np.array([])
        self.max_per_file = max_per_file
        self.save_path = save_path
        self.current_file_index = 0

        print(f"Creating {save_path}")
        if not os.path.exists(os.path.abspath(f"{save_path}")):
            os.makedirs(os.path.abspath(f"{save_path}"))

    def add_latent(self, latents: np.array, labels: np.array):
        self.latents = np.append(self.latents, latents, axis=0)
        self.labels = np.append(self.labels, labels)

        if self.latents.shape[0] >= self.max_per_file:
            self.save_next()

    def save_next(self):
        file_path = os.path.join(self.save_path, f"{(self.current_file_index):>08}.npz")
        np.savez(
            file_path,
            latents=self.latents[: self.max_per_file],
            labels=self.labels[: self.max_per_file],
        )
        self.current_file_index += 1
        self.latents = (
            np.empty((0, 32, 32))
            if self.latents.shape[0] == self.max_per_file
            else self.latents[self.max_per_file :]
        )
        self.labels = (
            np.array([])
            if self.labels.shape[0] == self.max_per_file
            else self.labels[self.max_per_file :]
        )

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


class LatentDataset(Dataset):
    def __init__(self, rootdir: str):
        self.rootdir = rootdir
        self.file_names = sorted(os.listdir(self.rootdir))

        # use first file size for "data_per_file"
        with np.load(f"{self.rootdir}/{self.file_names[0]}") as npzfile:
            self.data_per_file = npzfile["labels"].shape[0]

        # Account for the last file not necessarily having `data_per_file` samples
        self.len = (len(self.file_names) - 1) * self.data_per_file
        with np.load(f"{self.rootdir}/{self.file_names[-1]}") as npzfile:
            self.len += npzfile["labels"].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        with np.load(
            f"{self.rootdir}/{self.file_names[index // self.data_per_file]}"
        ) as npzfile:
            latents = npzfile["latents"][index % self.data_per_file][None, :, :]
            labels = npzfile["labels"][index % self.data_per_file]

        return (
            torch.tensor(latents, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
