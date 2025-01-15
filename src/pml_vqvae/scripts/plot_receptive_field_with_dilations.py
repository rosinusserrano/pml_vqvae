import torch
import matplotlib.pyplot as plt


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
    all_indices = [(50, 50)]
    current_indices = [(50, 50)]
    for d in reversed(dilations):
        next_indices = []
        for i in current_indices:
            next_indices.extend(get_indices_of_neighborhood(i, d))
        all_indices.extend(next_indices)
        current_indices = next_indices
    
    img = torch.zeros((100, 100), dtype=torch.long)

    for idx in all_indices:
        x, y = idx
        img[x, y] = torch.clamp(img[x, y], 10, 255)


if __name__ == "__main__":
    plot_receptive_field
