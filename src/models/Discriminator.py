from torch import nn
class DownBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(DownBlock, self).__init__()

        self.net = nn.Sequential(     
            nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=2, padding=1, bias=False),   nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x) # skip connection

class ConvBlock(nn.Module):
    def __init__(self, nbr_channels=64):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(     
            nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=False),   nn.BatchNorm2d(nbr_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, nbr_channels=64):
        super(Discriminator, self).__init__()

        self.entry_block = nn.Sequential(nn.Conv2d(3, nbr_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        self.conv_blocks = nn.Sequential(   DownBlock(in_channels=nbr_channels, out_channels=nbr_channels),
                                            ConvBlock(nbr_channels=nbr_channels),
                                            DownBlock(in_channels=nbr_channels, out_channels=nbr_channels*2),
                                            ConvBlock(nbr_channels=nbr_channels*2),
                                            DownBlock(in_channels=nbr_channels*2, out_channels=nbr_channels*4),
                                            ConvBlock(nbr_channels=nbr_channels*4),
                                            DownBlock(in_channels=nbr_channels*4, out_channels=nbr_channels*8),
                                            nn.Conv2d(nbr_channels*8, 1, kernel_size=3, stride=1, padding=1))

        self.pool = nn.AdaptiveAvgPool2d((32, 32))

        self.end_block = nn.Sequential( nn.Linear(1024, 32),
                                        nn.Dropout(0.5),
                                        nn.LeakyReLU(),
                                        nn.Linear(32, 1)) # do not use sigmoid which is in BCEWithLogitsLoss loss function

    def forward(self, x):
        x = self.entry_block(x)
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.end_block(x.view(x.size(0), 1024))
        return x