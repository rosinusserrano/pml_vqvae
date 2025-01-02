import torch
from torchvision import transforms
from PIL import Image
import os
from torcheval.metrics import FrechetInceptionDistance
import json

from pml_vqvae.dataset.dataloader import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    generated_images_dir = "artifacts/samples/"
    samples_per_class = 5

    transform_1 = transforms.Compose(
        [
            # transforms.Pad((85, 85, 86, 86)),
            transforms.Resize(299),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        ]
    )

    transform_2 = transforms.Compose(
        [
            # transforms.Pad((85, 85, 86, 86)),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        ]
    )

    fid = FrechetInceptionDistance(device=DEVICE)

    # save all pt paths in the directory
    pt_files = []
    for root, dirs, files in os.walk(generated_images_dir):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))

    pt_files = sorted(pt_files)

    fids = {}
    for i, gen_pt in enumerate(pt_files):
        class_idx = os.path.basename(gen_pt).split(".")[0]
        class_idx = int(class_idx)

        _, testloader = load_data(
            "imagenet",
            n_train=0,
            n_test=samples_per_class * 1000,
            class_idx=[i],
            batch_size=samples_per_class,
            shuffle=False,
        )

        t, _ = next(iter(testloader))
        t = (t + 1) / 2
        t = transform_2(t).to(DEVICE)
        fid.update(t, is_real=True)

        gen = torch.load(gen_pt).to(DEVICE)
        fid.update(gen, is_real=False)

        fid_value = fid.compute()

        print(f"Class: {class_idx}, FID: {fid_value.item()}")

        fids[i] = fid_value.item()

        fid.reset()

    # save the FID values as json
    out = os.path.join(generated_images_dir, "fids.json")
    with open(out, "w") as f:
        json.dump(fids, f)
