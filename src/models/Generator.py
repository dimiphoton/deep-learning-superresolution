from math import sqrt
from torch import nn
from .ResidualBlock import ResidualBlock
import math

class GeneratorV0(nn.Module):
    def __init__(self, nbr_channels=64, nbr_blocks=3, normalize=True):
        super(GeneratorV0, self).__init__()

        self.entry_block = nn.Sequential(nn.Conv2d(3, nbr_channels, kernel_size=9, stride=1, padding=4, padding_mode="replicate"), nn.PReLU())

        self.residual_blocks = nn.Sequential(*[ResidualBlock(nbr_channels=nbr_channels, normalize=normalize) for _ in range(nbr_blocks)])

        self.upscale_block = nn.Sequential( nn.Conv2d(nbr_channels, nbr_channels*4, kernel_size=3, stride=1, padding=1),
                                            nn.PixelShuffle(2),
                                            nn.PReLU())

        self.end_block = nn.Sequential(
                    nn.Conv2d(nbr_channels, int(sqrt(nbr_channels*3)), kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                    nn.PReLU(),
                    nn.Conv2d(int(sqrt(nbr_channels*3)), 3, kernel_size=9, stride=1, padding=4, padding_mode="replicate")) 

    def forward(self, x):
        x = self.entry_block(x)
        x = self.residual_blocks(x) + x
        x = self.upscale_block(x)
        x = self.end_block(x)
        return x

class UpscaleBlock(nn.Module):
    def __init__(self, nbr_channels=64):
        super(UpscaleBlock, self).__init__()

        self.net = nn.Sequential( nn.Conv2d(nbr_channels, nbr_channels*4, kernel_size=3, stride=1, padding=1),
                                            nn.PixelShuffle(2),
                                            nn.PReLU())

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, nbr_channels=64, nbr_blocks=3, normalize=True, scaling_factor=2):
        super(Generator, self).__init__()

        self.entry_block = nn.Sequential(nn.Conv2d(3, nbr_channels, kernel_size=9, stride=1, padding=4, padding_mode="replicate"), nn.PReLU())

        self.residual_blocks = nn.Sequential(*[ResidualBlock(nbr_channels=nbr_channels, normalize=normalize) for _ in range(nbr_blocks)])

        self.upscale_block = nn.Sequential(*[UpscaleBlock(nbr_channels=nbr_channels) for _ in range(int(math.log(scaling_factor, 2)))])

        self.end_block = nn.Sequential(
                    nn.Conv2d(nbr_channels, int(sqrt(nbr_channels*3)), kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                    nn.PReLU(),
                    nn.Conv2d(int(sqrt(nbr_channels*3)), 3, kernel_size=9, stride=1, padding=4, padding_mode="replicate")) 

    def forward(self, x):
        x = self.entry_block(x)
        x = self.residual_blocks(x) + x
        x = self.upscale_block(x)
        x = self.end_block(x)
        return x
