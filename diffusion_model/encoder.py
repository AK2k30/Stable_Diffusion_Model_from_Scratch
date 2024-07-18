"""
The VAE_Encoder class is a PyTorch module that serves as the encoder component of a Variational Autoencoder (VAE) model. It takes an input tensor of shape (batch_size, 3, height, width) and applies a series of convolutional, residual, and attention blocks to produce a latent representation of the input.

The forward method of the VAE_Encoder class takes an input tensor `x` and a noise tensor `noise`, and returns the latent representation of the input. The latent representation is obtained by passing the input through the various layers of the encoder, and then splitting the output into the mean and log-variance of the latent distribution. The final output is obtained by sampling from this latent distribution using the reparameterization trick.
"""
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            VAE_ResidualBlock(128, 128),

            VAE_ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(128, 256),

            VAE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),

            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        x = mean + stdev * noise

        x *= 0.18215

        return x
