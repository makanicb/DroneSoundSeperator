# src/model.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2d -> ReLU -> Conv2d -> ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class UNetSeparator(nn.Module):
    """U-Net architecture for magnitude mask prediction."""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Final output layer
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x shape: (batch, channels, freq_bins, time_steps)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self._downsample(e1))
        e3 = self.enc3(self._downsample(e2))

        # Bottleneck
        b = self.bottleneck(self._downsample(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output mask
        mask = torch.sigmoid(self.out_conv(d1))

        return mask

    def _downsample(self, x):
        return F.max_pool2d(x, 2)


