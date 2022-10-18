""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, init_features)
        self.down1 = Down(init_features, init_features * 2)
        self.down2 = Down(init_features * 2, init_features * 4)
        self.down3 = Down(init_features * 4, init_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(init_features * 8, (init_features * 16) // factor)
        self.up1 = Up(init_features * 16, (init_features * 8) // factor, bilinear)
        self.up2 = Up(init_features * 8, (init_features * 4) // factor, bilinear)
        self.up3 = Up(init_features * 4, (init_features * 2) // factor, bilinear)
        self.up4 = Up(init_features * 2, init_features, bilinear)
        self.outc = OutConv(init_features, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
