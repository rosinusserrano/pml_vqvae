"""Abstract class defining functions used for all models across this project"""

from abc import ABC, abstractmethod

import torch


# Just an interface to make sure all models have some methods
class PML_model(torch.nn.Module, ABC):
    def name(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loss_fn(model_outputs, target):
        raise NotImplementedError(
            "The fist argument must be the output(s) of the model and the second the target. The rest is free."
        )

    @abstractmethod
    def backward(self, loss):
        raise NotImplementedError(
            "just a wrapper function for the backward method as it will get more complicated for the vq-vae."
        )

    @staticmethod
    @abstractmethod
    def collect_stats(output, target, loss) -> dict[str, float]:
        raise NotImplementedError(
            """This function should output a dict where each key should map to a float"""
        )

    @staticmethod
    @abstractmethod
    def visualize_output(
        batch, output, target, prefix: str = "", base_dir: str = "."
    ) -> None:
        raise NotImplementedError(
            """This function should visualize the model output and save it to a file"""
        )
