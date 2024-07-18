import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
The VAE_AttentionBlock class is a PyTorch module that applies self-attention to the input tensor. It is used as part of the decoder in a variational autoencoder (VAE) model.

The module consists of a GroupNorm layer, a SelfAttention layer, and a residual connection. The input tensor is first reshaped and transposed, then passed through the SelfAttention layer. The output is then transposed and reshaped back to the original shape, and added to the original input via the residual connection.

This attention block is designed to capture long-range dependencies in the input data, which can be useful for tasks such as image generation or sequence modeling.
"""
class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residue = x

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residue

        return x

"""
    The VAE_ResidualBlock class is a PyTorch module that applies a residual connection to a sequence of convolutional and normalization layers. It is used as part of the decoder in a variational autoencoder (VAE) model.

    The module consists of two GroupNorm layers, two convolutional layers, and a residual connection. The input tensor is first passed through the first GroupNorm and convolutional layer, then the second GroupNorm and convolutional layer. The output is then added to the original input via the residual connection.

    This residual block is designed to help the model learn more complex representations by allowing the gradients to flow more easily through the network during training.
"""
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.groupnorm_1 = nn.GroupNorm(32, in_channels) 
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self. groupnorm_2(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)

"""
    The VAE_Decoder class is a PyTorch module that implements the decoder component of a Variational Autoencoder (VAE) model. The decoder takes a latent representation as input and generates an output image.

    The decoder consists of a series of convolutional, upsampling, and residual blocks that progressively increase the spatial resolution and refine the output image. The key components of the decoder are:

    - An initial 1x1 convolution to adjust the number of channels.
    - A series of VAE_ResidualBlock layers that apply residual connections to help the model learn complex representations.
    - A VAE_AttentionBlock layer that applies self-attention to the feature maps.
    - Upsampling layers to increase the spatial resolution of the output.
    - A final convolutional layer to produce the output image.

    The forward method of the VAE_Decoder class applies the sequence of layers to the input tensor and returns the generated output image.
"""
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215

        for module in self:
            x = module(x)

        return x




