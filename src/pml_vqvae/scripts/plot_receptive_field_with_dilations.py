import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def get_indices_of_neighborhood(index: tuple[int, int], dilation: int):
    x, y = index
    tl = (x - dilation, y - dilation)
    t = (x, y - dilation)
    tr = (x + dilation, y - dilation)
    cl = (x - dilation, y)
    cr = (x + dilation, y)
    bl = (x - dilation, y + dilation)
    b = (x, y + dilation)
    br = (x + dilation, y + dilation)

    return [tl, t, tr, cl, cr, bl, b, br]


def plot_receptive_field(dilations: list[int]):
    all_indices = [(20, 20)]
    current_indices = [(20, 20)]
    for d in tqdm(reversed(dilations)):
        next_indices = []
        for i in current_indices:
            next_indices.extend(get_indices_of_neighborhood(i, d))
        all_indices.extend(next_indices)
        current_indices = next_indices

    img = torch.zeros((32, 32), dtype=torch.long)

    for idx in all_indices:
        x, y = idx
        if x < 0 or y < 0 or x >= 32 or y >= 32:
            continue
        img[x, y] += 1
    img[img > 0] = torch.clamp(img[img > 0], 10, 255)

    dilations_string = "-".join(map(str, dilations))
    filename = f"drf-{dilations_string}.png"
    plt.imsave(filename, img)


if __name__ == "__main__":
    plot_receptive_field([1, 1, 2, 2, 3, 3, 4, 4])
