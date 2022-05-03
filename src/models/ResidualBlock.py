from torch import nn
class ResidualBlock(nn.Module):
    def __init__(self, nbr_channels=64, normalize=True):
        super(ResidualBlock, self).__init__()

        if normalize:
            self.net = nn.Sequential(     
                nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=False), nn.BatchNorm2d(nbr_channels), nn.PReLU(),
                nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=False), nn.BatchNorm2d(nbr_channels))
        else:
            # self.net = nn.Sequential(     
            #     nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=False), nn.BatchNorm2d(nbr_channels), nn.PReLU(),
            #     nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=False))
            self.net = nn.Sequential(     
                nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU(),
                nn.Conv2d(nbr_channels,  nbr_channels, kernel_size=3,  stride=1, padding=1, bias=True))

    def forward(self, x):
        return x + self.net(x) # skip connection