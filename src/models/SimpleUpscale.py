from torch import nn
class SimpleUpscale(nn.Module):
    def __init__(self):
        super(SimpleUpscale, self).__init__()

        #upsamplin vs conv transpose

        self.net = nn.Sequential(     
                     nn.Conv2d(3,   12, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(12), nn.LeakyReLU(),
                     nn.Conv2d(12,  12, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(12), nn.LeakyReLU(),
                     nn.Conv2d(12,  48, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(48), nn.LeakyReLU(),
                     nn.Conv2d(48,  48, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(48), nn.LeakyReLU(),
            nn.ConvTranspose2d(48,  12, kernel_size=2,  stride=2),              nn.BatchNorm2d(12), nn.LeakyReLU(),
                     nn.Conv2d(12,  12, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(12), nn.LeakyReLU(),
                     nn.Conv2d(12,   3, kernel_size=3,  stride=1, padding=1),   nn.BatchNorm2d(3), nn.LeakyReLU(),
                     nn.Conv2d(3,    3, kernel_size=3,  stride=1, padding=1))

    def forward(self, x):
        return self.net(x)