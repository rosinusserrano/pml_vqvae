import torch
import os
from pml_vqvae.baseline.pml_model_interface import PML_model
from pml_vqvae.visuals import show


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask, *args, **kwargs):

        p = kwargs["dilation"] * (kwargs["kernel_size"] - 1) // 2
        padding = (p, p)
        super().__init__(padding=padding, *args, **kwargs)
        self.register_buffer("mask", mask[None, None])

    def forward(self, x: torch.Tensor):
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x


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
        x = super(MaskedConv2d, self).forward(x)

        if cond_embedding != None:
            x += self.embed_matcher(cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )

        return x

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
        x = super(MaskedConv2d, self).forward(x)

        if cond_embedding != None:
            x += self.embed_matcher(cond_embedding).view(
                -1, 1, self.latent_shape[0], self.latent_shape[1]
            )

        return x

    def create_mask(self, mask_type: str, k: int):
        mask = torch.zeros(k, k)

        # set all to the left of center point
        mask[:, : k // 2] = 1

        # if we use the center pixel
        if mask_type == "B":
            mask[:, k // 2] = 1

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

        self.embed_matcher = torch.nn.Linear(
            num_classes, latent_shape[0] * latent_shape[1], bias=False
        )

    def forward(self, v_stack, h_stack, class_cond_embedding):
        # vertical stack
        v_stack_feat = self.conv_vertical(v_stack)  # [B, C, 28, 28]

        embed = self.embed_matcher(class_cond_embedding).view(
            -1, 1, self.latent_shape[0], self.latent_shape[1]
        )

        # add class conditional embedding
        conditioned_v_stack = v_stack_feat + embed
        # (C, num_classes)

        # split up features
        v_val, v_gate = torch.chunk(conditioned_v_stack, 2, dim=1)

        # apply activation and merge
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # horizontal stack
        h_stack_feat = self.conv_horizontal(h_stack)
        from_v_stack = self.conv_vert2horiz(v_stack_feat)

        h_stack_feat = h_stack_feat + from_v_stack

        # add class conditional embedding
        conditioned_h_stack = h_stack_feat + self.embed_matcher(
            class_cond_embedding
        ).view(-1, 1, self.latent_shape[0], self.latent_shape[1])

        # split up features
        h_val, h_gate = torch.chunk(conditioned_h_stack, 2, dim=1)

        # apply activation and merge
        h_stack_out = torch.tanh(h_val) * torch.sigmoid(h_gate)

        # apply 1x1 convolution
        h_stack_out = self.conv_horiz1x1(h_stack_out)

        # add residual connection
        h_stack_out += h_stack

        return v_stack_out, h_stack_out


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
        ],  # dilations for the masked convolutions, it also defines the number of layers
    ):
        super().__init__()

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

        # too optain a probability distribution over the codes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, class_idx: int):
        # get the embedding for the specific class
        cond_embedding = self.embedding(torch.tensor([class_idx]))  # (1,10)

        v_stack = self.v_stack(x, cond_embedding)  # [B, C, 30, 30]
        h_stack = self.h_stack(x, cond_embedding)  # [B, C, 30, 30]

        for layer in self.layers:
            v_stack, h_stack = layer(v_stack, h_stack, cond_embedding)

        out = self.conv_out(torch.relu(h_stack))

        # apply softmax to get a probability distribution of the codes
        x = self.softmax(out)

        return x

    def loss_fn(self, model_outputs, target):
        return torch.nn.functional.cross_entropy(model_outputs, target)

    def backward(self, loss: torch.Tensor):
        return loss.backward()

    @torch.no_grad()
    def sample(self, class_idx_list: list):
        # TODO: implement sampling
        pass

    @staticmethod
    def visualize_output(batch, output, target, prefix: str = "", base_dir: str = "."):
        show(batch, outfile=os.path.join(base_dir, f"{prefix}_original.png"))
        show(
            output,
            outfile=os.path.join(base_dir, f"{prefix}_reconstruction.png"),
        )

    def name(self):
        return "PixelCNN"


def train(train_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        for batch, target in train_loader:

            optimizer.zero_grad()

            batch = model.embed(batch, 0)
            output = model(batch)

            loss = model.loss_fn(output, target)
            model.backward(loss)
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    model = PixelCNN()
    print(model)
    input = torch.randn(2, 1, 32, 32)
    output = model(input, 0)
    print(output.shape)
    print(output)
    print("Done")
