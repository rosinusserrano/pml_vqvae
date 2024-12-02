"Functions to visualize stuff"

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def show(x: torch.Tensor, outfile: str = None, imgs_per_row: int = 8):
    """Plot a grid of images.

    `x`: A `torch.Tensor` of shape (batch_size x n_channels x
    height x width)

    `outfile` (optional): The path to the file where the plot
    should be saved to. If not specified, the plot is just
    shown directly (probably not possible on the cluster)

    `rows` (optional): The number of rows of the grid onto
    which to plot the images. Only has impact if `cols` is also
    specified, otherwise will simply figure out the optimal
    number of rows and cols.

    `cols` (optional): Same as `rows`.
    """
    assert len(x.shape) == 4, "Input should be batch with dimensions BS x C x H x W"

    image_grid = make_grid(x, nrow=imgs_per_row, padding=0, pad_value=1)

    # PyTorch uses the format C x H x W for images while
    # matplotlib uses H x W x C. Thus, we have to transpose
    # it accordingly
    image_grid = image_grid.permute(1, 2, 0)

    # Show the images
    plt.imshow(image_grid)

    # Remove tick labels and black border
    plt.axis("off")

    # If we specified a directory to save the plot save it,
    # otherwise simply show it.
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()
