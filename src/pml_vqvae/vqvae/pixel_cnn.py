import torch
import os
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show
from torchvision.transforms import v2
import torchvision
from torch.nn import functional as F

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
        num_classes: int,
        mask_type: str = "B",
        latent_shape: tuple = None,
        *args,
        **kwargs,
    ):

        mask = self.create_mask(mask_type, k=kwargs["kernel_size"])
        super().__init__(mask, *args, **kwargs)
        self.latent_shape = latent_shape

        if latent_shape:
            self.embed_matcher = torch.nn.Linear(
                num_classes, latent_shape[0] * latent_shape[1], bias=False
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
        num_classes: int,
        mask_type: str = "B",
        latent_shape: tuple = None,
        *args,
        **kwargs,
    ):

        mask = self.create_mask(mask_type, k=kwargs["kernel_size"])
        super().__init__(mask, *args, **kwargs)
        self.latent_shape = latent_shape

        if latent_shape:
            self.embed_matcher = torch.nn.Linear(
                num_classes, latent_shape[0] * latent_shape[1], bias=False
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
    def __init__(self, num_classes: int, latent_shape: tuple, *args, **kwargs):
        super().__init__()
        channels = kwargs["channels"]
        self.latent_shape = latent_shape

        # remove channels from kwargs
        kwargs.pop("channels")

        self.conv_vertical = VerticalStack(
            num_classes,
            in_channels=channels,
            out_channels=2 * channels,
            *args,
            **kwargs,
        )
        self.conv_horizontal = HorizontalStack(
            num_classes,
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

        self.h_embed_matcher = torch.nn.Linear(
            num_classes, latent_shape[0] * latent_shape[1], bias=False
        )

        self.v_embed_matcher = torch.nn.Linear(
            num_classes, latent_shape[0] * latent_shape[1], bias=False
        )

    def forward(self, v_stack, h_stack, class_cond_embedding):
        # vertical stack
        v_stack_feat = self.conv_vertical(v_stack)  # [B, C, 28, 28]

        v_embed = self.v_embed_matcher(class_cond_embedding).view(
            -1, 1, self.latent_shape[0], self.latent_shape[1]
        )

        # add class conditional embedding
        conditioned_v_stack = v_stack_feat + v_embed
        # (C, num_classes)

        # split up features
        v_val, v_gate = torch.chunk(conditioned_v_stack, 2, dim=1)

        # apply activation and merge
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # horizontal stack
        h_stack_feat = self.conv_horizontal(h_stack)
        from_v_stack = self.conv_vert2horiz(v_stack_feat)

        h_stack_feat = h_stack_feat + from_v_stack

        h_embed = self.h_embed_matcher(class_cond_embedding).view(
            -1, 1, self.latent_shape[0], self.latent_shape[1]
        )

        # add class conditional embedding
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


class PixelCNN(PML_model):
    def __init__(
        self,
        hidden_chan: int = 128,
        num_codes: int = 512,  # will be the output size
        num_classes: int = 10,  # number of classes in the dataset
        input_shape: tuple = (32, 32),  # latent shape of vqvae
        dilations: list = [
            1,
            2,
            1,
            4,
            1,
            2,
            1,
            2,
            1,
        ],  # dilations for the masked convolutions, it also defines the number of layers
    ):
        super().__init__()
        self.input_shape = input_shape

        # class conditional embedding
        self.embedding = torch.nn.Embedding(num_classes, num_classes, max_norm=1.0)

        self.v_stack = VerticalStack(
            dilation=1,
            num_classes=num_classes,
            latent_shape=input_shape,
            mask_type="A",  # don't use the center pixel only for very first layer
            in_channels=1,
            out_channels=hidden_chan,
            kernel_size=3,
        )
        self.h_stack = HorizontalStack(
            dilation=1,
            num_classes=num_classes,
            latent_shape=input_shape,
            mask_type="A",  # don't use the center pixel only for very first layer
            in_channels=1,
            out_channels=hidden_chan,
            kernel_size=3,
        )

        self.layers = torch.nn.ModuleList(
            [
                CondGatedMaskedConv2d(
                    num_classes=num_classes,
                    latent_shape=input_shape,
                    channels=hidden_chan,
                    kernel_size=3,
                    dilation=dil,
                )
                for dil in dilations
            ]
        )

        self.conv_out = torch.nn.Conv2d(
            in_channels=hidden_chan, out_channels=num_codes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor, class_idx: torch.Tensor):
        x = x * 2 - 1

        # get the embedding for the specific class
        cond_embedding = self.embedding(class_idx)  # (1,10)

        v_stack = F.elu(self.v_stack(x, cond_embedding))  # [B, C, 30, 30]
        h_stack = F.elu(self.h_stack(x, cond_embedding))  # [B, C, 30, 30]

        for layer in self.layers:
            v_stack, h_stack = layer(v_stack, h_stack, cond_embedding)

        out = self.conv_out(F.elu(h_stack))

        return out

    def loss_fn(self, model_outputs, target):
        return F.cross_entropy(model_outputs, torch.squeeze(target * 255).long())

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @torch.no_grad()
    def sample(self, class_idx_list: torch.Tensor):

        shape = (len(class_idx_list), 1, *self.input_shape)

        # Create empty image
        imgs = torch.zeros(shape, dtype=torch.float32).to(DEVICE)

        # Generation loop
        for h in range(self.input_shape[0]):
            for w in range(self.input_shape[1]):
                probs = F.softmax(self.forward(imgs, class_idx_list), dim=1)[:, :, h, w]
                tmp = torch.multinomial(probs, num_samples=1)
                imgs[:, :, h, w] = tmp / 255.0

        return imgs.cpu()

    @staticmethod
    def visualize_output(batch, output, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def training(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-3,
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            batch_losses = []
            for batch, labels in loader:

                optimizer.zero_grad()

                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)

                output = model(batch, labels)

                loss = model.loss_fn(output, batch)

                model.backward(loss)
                optimizer.step()

                with torch.no_grad():
                    batch_losses.append(loss.item())

            with torch.no_grad():
                probs = F.softmax(output, dim=1).cpu()
                output_img = torch.argmax(probs, dim=1, keepdim=True)
                model.visualize_output(batch.cpu(), output_img, prefix=f"train_{epoch}")
                epoch_loss = sum(batch_losses) / len(batch_losses)
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

    def name(self):
        return "PixelCNN"


if __name__ == "__main__":
    model = PixelCNN(
        input_shape=(28, 28), num_codes=256, hidden_chan=256, num_classes=10
    )

    model.to(DEVICE)

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    mnist = torchvision.datasets.MNIST(
        "./",
        transform=transforms,
        download=True,
    )

    loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )

    model.training(loader, epochs=10, learning_rate=1e-3)

    imgs = model.sample(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, dtype=torch.int).to(DEVICE)
    )

    show(imgs, outfile="samples.png", imgs_per_row=5)
