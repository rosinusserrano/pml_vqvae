"Functions to visualize stuff"

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json


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


def violin_plot(data: list, labels: list, title: str):
    """Plot a violin plot of the data.

    `data`: A list of lists of data points. Each list of data
    points will be plotted as a separate violin.
    """

    filename = title.lower().replace(" ", "_") + ".svg"

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    vp = plt.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        positions=range(len(data)),
        # quantiles=[[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]],
    )

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i % len(colors)])

    plt.xlabel("Data")
    plt.xticks(range(len(data)), labels)
    plt.ylabel("Error")
    plt.title(title)
    plt.savefig(filename)


if __name__ == "__main__":
    # data1 = np.random.normal(0, 1, 100)
    # data2 = np.random.normal(2, 1, 100)

    json_path = "artifacts/samples/fids.json"

    # read json file
    with open(json_path, "r") as f:
        data = json.load(f)

    values = list(data.values())

    violin_plot([values], ["A"], "Title")

    # from pytorch_fid import fid_score

    # real_images_dir = "artifacts"
    # fake_images_dir = "artifacts"

    # # Calculate the FID score between the two directories of images
    # fid_value = fid_score.calculate_fid_given_paths(
    #     [real_images_dir, fake_images_dir], batch_size=50, device="cpu", dims=2048
    # )

    # print(f"FID Score: {round(fid_value, 2)}")
