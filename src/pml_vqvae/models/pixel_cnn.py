from random import random
import torch
from dataclasses import dataclass, field
import yaml
import os
from pml_vqvae.models.pml_model_interface import PML_model
from pml_vqvae.models.vqvae import VQVAEConfig, VQVAE
from pml_vqvae.visuals import show
from torchvision.transforms import v2
import torchvision
from torch.nn import functional as F
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask, *args, **kwargs):

        p = kwargs["dilation"] * (kwargs["kernel_size"] - 1) // 2
        super().__init__(padding=(p, p), *args, **kwargs)
        self.register_buffer("mask", mask[None, None])

    def forward(self, x: torch.Tensor):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class VerticalStack(MaskedConv2d):
    def __init__(
        self,
        cond_embed_dim: int | None,
        mask_type: str = "B",
        latent_shape: tuple = None,
        *args,
        **kwargs,
    ):

        mask = self.create_mask(mask_type, k=kwargs["kernel_size"])
        super().__init__(mask, *args, **kwargs)
        self.latent_shape = latent_shape

        if latent_shape is not None and cond_embed_dim is not None:
            self.embed_matcher = torch.nn.Linear(
                cond_embed_dim, latent_shape[0] * latent_shape[1], bias=False
            )

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor = None):
        self.weight.data *= self.mask
        out = super(MaskedConv2d, self).forward(x)

        if cond_embedding != None:
            out += self.embed_matcher(cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )

        return out

    def create_mask(self, mask_type: str, k: int):

        mask = torch.zeros(k, k)

        # set all above center point
        mask[: k // 2] = 1

        # if we use the center pixel
        if mask_type == "B":
            mask[k // 2] = 1

        return mask


class HorizontalStack(MaskedConv2d):
    def __init__(
        self,
        cond_embed_dim: int | None,
        mask_type: str = "B",
        latent_shape: tuple = None,
        *args,
        **kwargs,
    ):

        mask = self.create_mask(mask_type, k=kwargs["kernel_size"])
        super().__init__(mask, *args, **kwargs)
        self.latent_shape = latent_shape

        if latent_shape is not None and cond_embed_dim is not None:
            self.embed_matcher = torch.nn.Linear(
                cond_embed_dim, latent_shape[0] * latent_shape[1], bias=False
            )

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor = None):
        self.weight.data *= self.mask
        out = super(MaskedConv2d, self).forward(x)

        if cond_embedding != None:
            out += self.embed_matcher(cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )

        return out

    def create_mask(self, mask_type: str, k: int):
        mask = torch.zeros(k, k)

        # set all to the left of center point
        mask[k // 2, : k // 2] = 1

        # if we use the center pixel
        if mask_type == "B":
            mask[k // 2, k // 2] = 1

        return mask


class CondGatedMaskedConv2d(torch.nn.Module):
    def __init__(
        self, cond_embed_dim: int | None, latent_shape: tuple, *args, **kwargs
    ):
        super().__init__()
        channels = kwargs["channels"]
        self.latent_shape = latent_shape

        # remove channels from kwargs
        kwargs.pop("channels")

        self.conv_vertical = VerticalStack(
            cond_embed_dim,
            in_channels=channels,
            out_channels=2 * channels,
            *args,
            **kwargs,
        )
        self.conv_horizontal = HorizontalStack(
            cond_embed_dim,
            in_channels=channels,
            out_channels=2 * channels,
            *args,
            **kwargs,
        )
        self.conv_vert2horiz = torch.nn.Conv2d(
            in_channels=2 * channels,
            out_channels=2 * channels,
            kernel_size=1,
            padding=0,
        )
        self.conv_horiz1x1 = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
        )

        if cond_embed_dim is not None:
            self.h_embed_matcher = torch.nn.Linear(
                cond_embed_dim, latent_shape[0] * latent_shape[1], bias=False
            )

            self.v_embed_matcher = torch.nn.Linear(
                cond_embed_dim, latent_shape[0] * latent_shape[1], bias=False
            )

    def forward(
        self,
        v_stack: torch.Tensor,
        h_stack: torch.Tensor,
        class_cond_embedding: torch.Tensor | None,
    ):
        # vertical stack
        v_stack_feat = self.conv_vertical(v_stack)  # [B, C, 28, 28]

        # add class conditional embeddings if conditioning is enabled
        if class_cond_embedding is not None:
            v_embed = self.v_embed_matcher(class_cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )
        else:
            v_embed = torch.zeros_like(v_stack_feat)
        conditioned_v_stack = v_stack_feat + v_embed

        # split up features
        v_val, v_gate = torch.chunk(conditioned_v_stack, 2, dim=1)

        # apply activation and merge
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # horizontal stack
        h_stack_feat = self.conv_horizontal(h_stack)
        from_v_stack = self.conv_vert2horiz(v_stack_feat)

        h_stack_feat = h_stack_feat + from_v_stack

        # class conditioning, same as for vertical stack above
        if class_cond_embedding is not None:
            h_embed = self.h_embed_matcher(class_cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )
        else:
            h_embed = torch.zeros_like(h_stack_feat)
        conditioned_h_stack = h_stack_feat + h_embed

        # split up features
        h_val, h_gate = torch.chunk(conditioned_h_stack, 2, dim=1)

        # apply activation and merge
        h_stack_out = torch.tanh(h_val) * torch.sigmoid(h_gate)

        # apply 1x1 convolution
        h_stack_out = self.conv_horiz1x1(h_stack_out)

        # add residual connection
        h_stack_out = h_stack_out + h_stack

        return F.elu(v_stack_out), F.elu(h_stack_out)


@dataclass
class PixelCNNConfig:
    """Config for PixelCNN."""

    hidden_chan: int = 128
    name: str = "PixelCNN"
    num_codes: int = 512  # will be the output size
    conditional: bool = True
    num_classes: int | None = 10  # number of classes in the dataset
    conditional_embedding_dim: int | None = 256
    input_shape: tuple = (32, 32)  # latent shape of vqvae
    # dilations for the masked convolutions, it also defines the number of layers
    dilations: list[int] | str = field(
        default_factory=lambda: [1, 2, 1, 4, 1, 2, 1, 2, 1]
    )
    vqvae_path: str | None = None
    use_one_hot: bool = False
    use_code_embeddings: bool = False


class PixelCNN(PML_model):
    def __init__(
        self,
        config: PixelCNNConfig,
    ):
        if config.conditional and (
            config.num_classes is None or config.conditional_embedding_dim is None
        ):
            raise ValueError(
                "If conditional, num_classes and conditional_embedding_dim have to be provided."
            )

        if config.use_code_embeddings:
            if config.vqvae_path is None:
                raise ValueError(
                    "Have to provide the path to the VQVAE if you want to use the code embeddings"
                )
            if config.use_one_hot:
                raise ValueError(
                    "Can't use both, decide for either one hot or code embeddings or none."
                )

        super().__init__()

        # Support dilations as string for hyperparameter optimization
        if isinstance(config.dilations, str):
            config.dilations = list(map(int, config.dilations.split("-")))

        self.config = config

        self.codebook = None
        self.decoder = None
        if self.config.vqvae_path is not None:
            vqvae = self.get_vqvae_model()
            self.codebook = vqvae.codebook.to(DEVICE)
            self.decoder = vqvae.decoder.requires_grad_(False).to(DEVICE)

        self.input_channels = 1
        if self.config.use_code_embeddings:
            self.input_channels = self.codebook.shape[1]
        if self.config.use_one_hot:
            self.input_channels = self.config.hidden_chan

        # class conditional embedding
        if config.conditional:
            self.embedding = torch.nn.Embedding(
                config.num_classes, config.conditional_embedding_dim
            )

        if config.use_one_hot:
            self.one_hot_conv = torch.nn.Conv2d(
                self.config.num_codes,
                self.config.hidden_chan,
                kernel_size=1,
            )

        self.v_stack = VerticalStack(
            dilation=config.dilations[0],
            cond_embed_dim=config.conditional_embedding_dim,
            latent_shape=config.input_shape,
            mask_type="A",  # don't use the center pixel only for very first layer
            in_channels=self.input_channels,
            out_channels=config.hidden_chan,
            kernel_size=3,
        )
        self.h_stack = HorizontalStack(
            dilation=config.dilations[0],
            cond_embed_dim=config.conditional_embedding_dim,
            latent_shape=config.input_shape,
            mask_type="A",  # don't use the center pixel only for very first layer
            in_channels=self.input_channels,
            out_channels=config.hidden_chan,
            kernel_size=3,
        )

        self.layers = torch.nn.ModuleList(
            [
                CondGatedMaskedConv2d(
                    cond_embed_dim=config.conditional_embedding_dim,
                    latent_shape=config.input_shape,
                    channels=config.hidden_chan,
                    kernel_size=3,
                    dilation=dil,
                )
                for dil in config.dilations[1:]
            ]
        )

        self.conv_out = torch.nn.Conv2d(
            in_channels=config.hidden_chan,
            out_channels=config.num_codes,
            kernel_size=1,
            padding=0,
        )

    def get_vqvae_model(self):
        config_file = f"{self.config.vqvae_path}/config.yaml"
        model_file = f"{self.config.vqvae_path}/model.pth"

        print("Reading config file")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        print("Loading model")
        model_config = VQVAEConfig(**config_dict["model_config"]["value"])
        vqvae = VQVAE(model_config)
        vqvae.load_state_dict(torch.load(model_file, weights_only=True))

        return vqvae

    def forward(self, x: torch.Tensor, class_idx: torch.Tensor | None = None):
        if self.config.conditional:
            # get the embedding for the specific class
            cond_embedding = self.embedding(class_idx)
        else:
            # else disable conditioning
            cond_embedding = None

        # Map indices in specified way (code embeddings or one-hot)
        if self.config.use_code_embeddings:
            x = self.codebook[x.long()].squeeze().permute(0, 3, 1, 2)
        if self.config.use_one_hot:
            x = (
                F.one_hot(x.long(), self.config.num_codes)
                .float()
                .squeeze()
                .permute(0, 3, 1, 2)
            )
            x = self.one_hot_conv(x)

        v_stack = F.elu(self.v_stack(x, cond_embedding))  # [B, C, 30, 30]
        h_stack = F.elu(self.h_stack(x, cond_embedding))  # [B, C, 30, 30]

        for layer in self.layers:
            v_stack, h_stack = layer(v_stack, h_stack, cond_embedding)

        out = self.conv_out(F.elu(h_stack))

        return out

    def loss_fn(self, model_outputs, target: torch.Tensor):
        target = torch.squeeze(target).long()
        loss = F.cross_entropy(model_outputs, target)
        self.batch_stats = {"Loss": loss.item()}
        return loss

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @torch.no_grad()
    def sample(
        self,
        class_idx_list: torch.Tensor | None = None,
        num_samples: int | None = None,
    ):
        if self.config.conditional and class_idx_list is None:
            raise ValueError(
                "Have to provide class idx list if pixelcnn is conditional."
            )

        if not self.config.conditional and num_samples is None:
            raise ValueError(
                "Unconditional PixelCNN needs the number of samples specified."
            )

        shape = (
            (len(class_idx_list), 1, *self.config.input_shape)
            if num_samples is None
            else (num_samples, 1, *self.config.input_shape)
        )

        # Create empty image
        imgs = torch.zeros(shape, dtype=torch.float32).to(DEVICE)

        # Generation loop
        for h in trange(self.config.input_shape[0]):
            for w in range(self.config.input_shape[1]):
                preds = self.forward(imgs, class_idx_list)

                probs = F.softmax(preds, dim=1)[:, :, h, w]
                tmp = torch.multinomial(probs, num_samples=1)

                imgs[:, :, h, w] = tmp

        return imgs.cpu()

    @torch.no_grad()
    def complete(
        self,
        incomplete: torch.Tensor,
        class_idx_list: torch.Tensor | None = None,
    ):
        if self.config.conditional and class_idx_list is None:
            raise ValueError(
                "Have to provide class idx list if pixelcnn is conditional."
            )

        # Add channel dimension if not there
        if len(incomplete.shape) == 3:
            incomplete = incomplete[:, None, :, :]

        # Generation loop
        for h in trange(self.config.input_shape[0]):
            for w in range(self.config.input_shape[1]):
                if incomplete[0, 0, h, w].item() >= 0:
                    continue

                preds = self.forward(incomplete, class_idx_list)

                probs = F.softmax(preds, dim=1)[:, :, h, w]
                tmp = torch.multinomial(probs, num_samples=1)

                incomplete[:, :, h, w] = tmp

        return incomplete.cpu()

    def visualize_output(self, output: torch.Tensor):
        return torch.argmax(output, dim=1, keepdim=True)

    def name(self):
        return "PixelCNN"
