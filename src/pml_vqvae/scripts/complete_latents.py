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
    codebook_size=512,
    commitment_weight=2,
    hidden_dimension=256,
    embedding_dimension=256,
    codebook_initialization_radius=0.5,
)
vqvae = VQVAE(vqvaeconfig).to(DEVICE)
vqvae.load_state_dict(
    torch.load("artifacts/konni_replacement_vqvae/model.pth", weights_only=True)
)

print("Loading PixelCNN")
pixelcnnconfig = PixelCNNConfig(
    num_codes=512,
    conditional=False,
    hidden_chan=128,
    num_classes=None,
    conditional_embedding_dim=None,
    dilations="1-1-1-1-1-1-1-1-1-1-1-1-1-1-1",
    input_shape=(32, 32),
)
pixelcnn = PixelCNN(pixelcnnconfig).to(DEVICE)
pixelcnn.load_state_dict(
    torch.load(
        "artifacts/hyperopt-XI-pixelcnn-unconditional_19/model_20.pth",
        weights_only=True,
    )
)

print("Sampling latent indices with PixelCNN")
indices = pixelcnn.sample(num_samples=64)

print("Generating images with decoder of VQVAE")
generated_images = vqvae.decode(indices.long()).detach().cpu()
show(generated_images, outfile="vqvae_generations_unconditional2.png")

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
