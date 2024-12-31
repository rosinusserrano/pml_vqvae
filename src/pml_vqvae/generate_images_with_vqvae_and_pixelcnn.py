import torch

from pml_vqvae.models.vqvae import VQVAE, VQVAEConfig
from pml_vqvae.models.pixel_cnn import PixelCNN, PixelCNNConfig
from pml_vqvae.visuals import show
from pml_vqvae.dataset.dataloader import load_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("On device", DEVICE)

print("Loading VQVAE")
vqvaeconfig = VQVAEConfig(
    codebook_size=1024,
    commitment_weight=2,
    hidden_dimension=64,
    embedding_dimension=64,
)
vqvae = VQVAE(vqvaeconfig).to(DEVICE)
vqvae.load_state_dict(
    torch.load("artifacts/vqvae_konni_easy_params/model.pth", weights_only=True)
)

print("Loading PixelCNN")
pixelcnnconfig = PixelCNNConfig(
    num_codes=1024,
    hidden_chan=256,
    num_classes=1000,
    input_shape=(32, 32),
)
pixelcnn = PixelCNN(pixelcnnconfig).to(DEVICE)
pixelcnn.load_state_dict(
    torch.load("artifacts/pixelcnn on latent test_26/model_10.pth", weights_only=True)
)

print("Sampling latent indices with PixelCNN")
indices = pixelcnn.sample(
    torch.ones((64,)).long().to(DEVICE), probabilistic_sampling_prob=0.8
)

print("Generating images with decoder of VQVAE")
generated_images = vqvae.decode(indices.long()).detach().cpu()
show(generated_images, outfile="vqvae_generations.png")

print("Getting imagenet for test batch reconstruction")
imgnet, _ = load_data("imagenet", batch_size=64)
batch, labels = next(iter(imgnet))
indices = vqvae.encode(batch.to(DEVICE))

print("Reconstructing images with decoder of VQVAE")
generated_images = vqvae.decode(indices.long()).detach().cpu()
show(generated_images, outfile="vqvae_reconstructions.png")

print("Generating images from random code indices")
indices = torch.randint(0, 1024, (64, 32, 32))
generated_images = vqvae.decode(indices.long()).detach().cpu()
show(generated_images, outfile="vqvae_generations_random.png")

print("Plotting original images")
show(batch, outfile="vqvae_original.png")
