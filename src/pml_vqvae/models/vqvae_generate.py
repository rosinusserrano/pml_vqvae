import torch
from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
import os
from PIL import Image
import numpy as np

from pml_vqvae.visuals import show

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate(
    vqvae_path: str,
    pixelcnn_path: str,
    sample_per_class: int = 5,
    classes: list = None,
    out_folder: str = "artifacts",
    save_as_png: bool = True,
):
    # Load the VQ-VAE model

    config = VQVAEConfig(
        codebook_size=1024,
        commitment_weight=2.2348288297357928,
        hidden_dimension=64,
        embedding_dimension=64,
    )

    vqvae = VQVAE(config)
    vqvae.load_state_dict(torch.load(vqvae_path, weights_only=True))
    vqvae.eval()
    vqvae.to(DEVICE)

    # Load the PixelCNN model

    config = PixelCNNConfig(
        hidden_chan=256,
        num_codes=1024,  # will be the output size
        num_classes=30,  # number of classes in the dataset
        input_shape=(32, 32),  # latent shape of vqvae
        dilations=[1, 2, 1, 4, 1, 2, 1, 2, 1],
    )

    pixelcnn = PixelCNN(config)
    pixelcnn.load_state_dict(torch.load(pixelcnn_path, weights_only=True))
    pixelcnn.eval()
    pixelcnn.to(DEVICE)

    out_folder = os.path.join(out_folder, "samples_3")
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if classes is None:
        classes = list(range(30))

    print("Generating samples...")
    for class_idx in classes:
        print("sample ...")
        latents = pixelcnn.sample(
            torch.tensor([class_idx] * sample_per_class, dtype=torch.int).to(DEVICE)
        )
        print("decode ...")
        generated = vqvae.decode(latents.to(DEVICE)).to("cpu")

        generated = (generated + 1) / 2

        generated = generated.clamp(0, 1)

        print("save ...")
        if save_as_png:
            # if not os.path.exists(os.path.join(out_folder, str(class_idx))):
            #     os.mkdir(os.path.join(out_folder, str(class_idx)))

            # for i, img in enumerate(generated):
            #     img_path = os.path.join(out_folder, str(class_idx), f"{i}.png")
            #     img = img.permute(1, 2, 0) * 255
            #     img = img.detach().numpy().astype(np.uint8)
            #     Image.fromarray(img).save(img_path)

            labels = {4: "hammerhead", 9: "ostrich"}

            show(
                generated,
                os.path.join(out_folder, f"{str(class_idx)}.svg"),
                imgs_per_row=5,
                title=labels[class_idx],
            )

        # torch.save(generated, os.path.join(out_folder, f"{str(class_idx)}.pt"))

        print(
            f"Generated {sample_per_class} samples for class {class_idx} in {out_folder}/{class_idx}"
        )


if __name__ == "__main__":
    vqvae_path = "vqvae.pth"
    pixelcnn_path = "pixelcnn.pth"
    generate(
        vqvae_path,
        pixelcnn_path,
        sample_per_class=25,
        classes=[4, 9],
        out_folder="artifacts",
        save_as_png=True,
    )
