import torch


# Just an interface to make sure all models have some methods
class PML_model(torch.nn.Module):
    def name(self):
        raise NotImplementedError

    def loss_fn(self):
        raise NotImplementedError(
            "The fist argument must be the output of the model and the second the target. The rest is free."
        )

    def backward(self, loss):
        raise NotImplementedError(
            "just a wrapper function for the backward method as it will get more complicated for the vq-vae."
        )
