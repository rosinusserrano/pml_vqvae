import torch
import numpy as np

from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
from pml_vqvae.visuals import show
from pml_vqvae.dataset.dataloader import load_data
from pml_vqvae.scripts.inspect_usage_of_codes import get_usage_of_codes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("On device", DEVICE)

print("Loading VQVAE")
vqvaeconfig = VQVAEConfig(
    codebook_size=1024,
    commitment_weight=2,
    hidden_dimension=256,
    embedding_dimension=256,
    codebook_initialization_radius=0.5,
)
vqvae = VQVAE(vqvaeconfig).to(DEVICE)
vqvae.load_state_dict(
    torch.load("artifacts/final_replacement_vqvae/model.pth", weights_only=True)
)

print("Loading PixelCNN")
pixelcnnconfig = PixelCNNConfig(
    num_codes=1024,
    conditional=True,
    hidden_chan=128,
    num_classes=1000,
    conditional_embedding_dim=64,
    # dilations=[1, 2, 1, 4, 1, 2, 1, 2, 1],
    input_shape=(32, 32),
    vqvae_path="artifacts/final_replacement_vqvae",
    use_code_embeddings=False,
    use_one_hot=True,
)
pixelcnn = PixelCNN(pixelcnnconfig).to(DEVICE)
pixelcnn.load_state_dict(
    torch.load(
        "artifacts/final_pixelcnn_ohe_conditional/model.pth",
        weights_only=True,
    )
)

# for clsidx in range():
print("Sampling latent indices with PixelCNN")
indices = pixelcnn.sample(class_idx_list=(torch.ones(64).long()).to(DEVICE))

print("Generating images with decoder of VQVAE")
generated_images = vqvae.decode(indices.long()).detach().cpu()
show(generated_images, outfile=f"vqvae_generations_cls1.png")

# print("Getting imagenet for test batch reconstruction")
# imgnet, _ = load_data("imagenet", batch_size=64)
# batch, labels = next(iter(imgnet))
# indices = vqvae.encode(batch.to(DEVICE))

# print("Reconstructing images with decoder of VQVAE")
# generated_images = vqvae.decode(indices.long()).detach().cpu()
# show(generated_images, outfile="vqvae_reconstructions.png")

# print("Generating images from random code indices")
# indices = torch.randint(0, 1024, (64, 32, 32))
# generated_images = vqvae.decode(indices.long()).detach().cpu()
# show(generated_images, outfile="vqvae_generations_random.png")

# print(
#     "Generating images from random code indices but restrict to the ones actually used in the latent dataset"
# )
# code_usage = get_usage_of_codes()
# np.random.choice(
#     a=np.arange(1024),
#     size=(64, 32, 32),
#     p=(code_usage / code_usage.sum()).detach().cpu().numpy(),
# )
# generated_images = vqvae.decode(indices.long()).detach().cpu()
# show(generated_images, outfile="vqvae_generations_random_but_used_indices.png")

# print("Plotting original images")
# show(batch, outfile="vqvae_original.png")
