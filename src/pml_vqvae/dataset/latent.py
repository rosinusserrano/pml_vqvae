from torch.utils.data import Dataset
import torch
import numpy as np
import os

MAX_PER_FILE = 10000


class LatentDataset(Dataset):
    def __init__(self):
        self.latents = np.empty((0, 32, 32))
        self.labels = np.array([])

    def add_latent(self, latents: np.array, labels: np.array):
        self.latents = np.append(self.latents, latents, axis=0)
        self.labels = np.append(self.labels, labels)
        print(f"Added {len(latents)} latents to dataset")

    def save(self, path: str, name: str = "latent_dataset"):
        folder_path = os.path.join(path, name)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Save dataset to file in chunks of not more than MAX_PER_FILE
        for i in range(0, len(self.latents), MAX_PER_FILE):
            file_path = os.path.join(folder_path, f"{name}_{i // MAX_PER_FILE}.npz")
            np.savez(
                file_path,
                latents=self.latents[i : i + MAX_PER_FILE],
                labels=self.labels[i : i + MAX_PER_FILE],
            )

        print(f"Saved dataset to {folder_path}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]
