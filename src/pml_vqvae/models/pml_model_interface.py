"""Abstract class defining functions used for all models across this project"""

import torch


# Just an interface to make sure all models have some methods
class PML_model(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_stats = {}
        self.train_stats = {}

    def name(self):
        raise NotImplementedError

    @staticmethod
    def loss_fn(model_outputs, target):
        raise NotImplementedError(
            "The fist argument must be the output(s) of the model and the second the target. The rest is free."
        )

    def backward(self, loss):
        raise NotImplementedError(
            "just a wrapper function for the backward method as it will get more complicated for the vq-vae."
        )

    def visualize_output(self, output) -> torch.Tensor:
        return output