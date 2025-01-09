"Functions to visualize stuff"

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches


def show(
    x: torch.Tensor, outfile: str = None, imgs_per_row: int = 8, title: str = None
):
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

    plt.title(title)

    # Remove tick labels and black border
    plt.axis("off")

    # If we specified a directory to save the plot save it,
    # otherwise simply show it.
    if outfile is not None:
        plt.savefig(
            outfile,
        )
    else:
        plt.show()


def violin_plot(data: list, labels: list, title: str, ylabel: str):
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

    patches = [mpatches.Patch(color=colors[i % len(colors)]) for i in range(len(data))]
    mean = [np.round(np.mean(d), 3) for d in data]
    l = [f"mean: {m}" for m in mean]

    vp = plt.violinplot(
        data,
        showmeans=True,
        showmedians=False,
        showextrema=True,
        positions=range(len(data)),
        # quantiles=[[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]],
    )

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i % len(colors)])

    plt.legend(patches, l)

    # plt.xlabel("Data")
    plt.xticks(range(len(data)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    # data1 = np.random.normal(0, 1, 100)
    # data2 = np.random.normal(2, 1, 100)

    json_path = "artifacts/samples/fids.json"
    json_path_2 = "artifacts/samples/fids_train_test.json"
    json_path_3 = "artifacts/samples/fids_random.json"

    # read json file
    with open(json_path, "r") as f:
        data1 = json.load(f)

    with open(json_path_2, "r") as f:
        data2 = json.load(f)

    with open(json_path_3, "r") as f:
        data3 = json.load(f)

    values1 = list(data1.values())
    print("mean: ", np.mean(values1))
    values2 = list(data2.values())
    print("mean: ", np.mean(values2))
    values3 = list(data3.values())
    print("mean: ", np.mean(values3))

    violin_plot(
        [values1, values2, values3],
        ["Generations", "Train/Test", "Random"],
        "FID Scores",
    )

    # from pytorch_fid import fid_score

    # real_images_dir = "artifacts"
    # fake_images_dir = "artifacts"

    # # Calculate the FID score between the two directories of images
    # fid_value = fid_score.calculate_fid_given_paths(
    #     [real_images_dir, fake_images_dir], batch_size=50, device="cpu", dims=2048
    # )

    # print(f"FID Score: {round(fid_value, 2)}")
