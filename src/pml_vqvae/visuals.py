"Functions to visualize stuff"
import math
import torch
import matplotlib.pyplot as plt


def show_image_grid(x: torch.Tensor,
                    outfile: str = None,
                    rows: int = None,
                    cols: int = None,
                    inch_per_pixel: float = 0.05):
    """Plot a grid of images.

    `x`: A `torch.Tensor` of shape (batch_size x n_channels x
    height x width)

    `outfile` (optional): The path to the file where the plot
    should be saved to. If not specified, the plot is just
    shown directly (probably not possible on the cluster)

    `rows` (optional): The number of rows of the grid onto
    which to plot the images. Only has impact if `cols` is also
    specified, otherwise will simply figure aout the optimal
    number of rows and cols.

    `cols` (optional): Same as `rows`.

    `inch_per_pixel` (optional): Defaults to 0.05
    """
    assert len(x.shape) == 4,\
        "Input should be batch with dimensions BS x C x H x W"
    n_images = x.shape[0]
    height, width = x.shape[2], x.shape[3]

    # If not specified; determine number of rows and columns
    # based on the numbers that factorize the number of
    # images and are closest together
    if rows is None or cols is None:
        # stolen from https://stackoverflow.com/questions/39248245/factor-an-integer-to-something-as-close-to-a-square-as-possible
        rows = math.ceil(math.sqrt(n_images))
        cols = int(n_images / rows)
        while cols * rows != float(n_images):
            rows -= 1
            cols = int(n_images / rows)

    # PyTorch uses the format C x H x W for images while
    # matplotlib uses H x W x C. Thus, we have to transpose
    # it accordingly
    x = x.permute(0, 2, 3, 1)

    # Create matplotlib figure and axes and set width and
    # height of figure if `figsize is specified`
    fig, axs = plt.subplots(rows,
                            cols,
                            squeeze=False,
                            gridspec_kw={
                                "wspace": 0,
                                "hspace": 0
                            })

    # A bit hacky but in order to eliminate the gaps between
    # the images in the grid, I had to set the aspect="auto"
    # argument in the axs.imshow() method, which stops
    # matplotlib from forcing them to be square thus I had
    # to set the figsize depending on the size of the grid
    # and it can be scaled using the `inch_per_pixel` param.
    fig.set_size_inches(cols * width * inch_per_pixel,
                        rows * height * inch_per_pixel)

    # Fill the grid with the images
    for i in range(rows * cols):
        img = x[i]
        row_i = i // cols
        col_i = i % cols
        axs[row_i, col_i].imshow(img.detach().cpu().numpy(), aspect="auto")
        axs[row_i, col_i].tick_params(left=False,
                                      right=False,
                                      labelleft=False,
                                      labelbottom=False,
                                      bottom=False)

    # If we specified a directory to save the plot save it,
    # otherwise simply show it.
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    images = torch.randn((32, 1, 8, 8))
    show_image_grid(images)
    show_image_grid(images, inch_per_pixel=0.1)
    show_image_grid(images, outfile="test.png")
