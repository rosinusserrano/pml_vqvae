import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show
from pml_vqvae.nnutils import ResidualBlock


class PixelCNN(PML_model):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return

    def loss_fn(self, model_outputs, target):
        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "PixelCNN"
