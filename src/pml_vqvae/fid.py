import torch
from torchvision import transforms
from PIL import Image
import os
from torcheval.metrics import FrechetInceptionDistance
import json

from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.visuals import show

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    generated_images_dir = "artifacts/samples/"
    samples_per_class = 50

    transform_1 = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.ToTensor(),
        ]
    )

    transform_2 = transforms.Compose(
        [
            transforms.Resize(299),
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

        trainloader, testloader = load_data(
            "imagenet",
            n_train=samples_per_class * 1000,
            n_test=samples_per_class * 1000,
            class_idx=[i],
            batch_size=samples_per_class,
            shuffle=False,
        )

        t, _ = next(iter(testloader))
        t = (t + 1) / 2
        t = transform_2(t).to(DEVICE)
        t = t.clamp(0, 1)

        # show(t.cpu(), "real.png")

        fid.update(t.to(DEVICE), is_real=True)

        # t, _ = next(iter(trainloader))
        # t = (t + 1) / 2
        # t = transform_2(t).to(DEVICE)
        # t = t.clamp(0, 1)

        # t = torch.rand((samples_per_class, 3, 299, 299))

        gen = torch.load(gen_pt, weights_only=True).to(DEVICE)

        # show(gen.cpu(), "fake.png")
        fid.update(gen.to(DEVICE), is_real=False)

        fid_value = fid.compute()

        print(f"Class: {class_idx}, FID: {fid_value.item()}")

        fids[i] = fid_value.item()

        fid.reset()

    # save the FID values as json
    out = os.path.join(generated_images_dir, "fids.json")
    with open(out, "w") as f:
        json.dump(fids, f)
