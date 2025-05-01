# src/multi_channel_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelConvBlock(nn.Module):
    """Convolutional block that preserves multi-channel information."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.2, use_batch_norm=True):
        super().__init__()
        
        layers = []
        # First convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Second convolution
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Dropout for regularization
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class MultiChannelEncoder(nn.Module):
    """Encoder that processes multi-channel audio."""
    def __init__(self, input_channels, base_channels=32, depth=3, use_batch_norm=True, dropout_rate=0.2):
        super().__init__()
        self.depth = depth
        self.encoder_blocks = nn.ModuleList()
        
        # Initial block that takes multiple channels
        self.encoder_blocks.append(
            MultiChannelConvBlock(
                input_channels, 
                base_channels, 
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
        )
        
        # Remaining encoder blocks
        for i in range(1, depth):
            in_channels = base_channels * (2**(i-1))
            out_channels = base_channels * (2**i)
            self.encoder_blocks.append(
                MultiChannelConvBlock(
                    in_channels, 
                    out_channels,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm
                )
            )

    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, 2)  # Downsample
        return features, x

class MultiChannelDecoder(nn.Module):
    """Decoder for multi-channel audio processing."""
    def __init__(self, base_channels=32, depth=3, use_batch_norm=True, dropout_rate=0.2):
        super().__init__()
        self.depth = depth
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Create upsampling and decoder blocks
        for i in range(depth-1, -1, -1):
            # Channels for each level
            in_channels = base_channels * (2**i)
            
            if i < depth-1:  # Not the bottleneck level
                in_channels *= 2  # Double for skip connection
                
            out_channels = base_channels * (2**max(0, i-1))
            if i == 0:  # Last level
                out_channels = base_channels
            
            # Upsampling block
            self.upsample_blocks.append(
                nn.ConvTranspose2d(in_channels // 2 if i < depth-1 else in_channels,
                                  out_channels, 
                                  kernel_size=2, stride=2)
            )
            
            # Decoder block
            self.decoder_blocks.append(
                MultiChannelConvBlock(
                    in_channels, 
                    out_channels,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm
                )
            )

    def forward(self, features, bottleneck):
        x = bottleneck
        
        for i in range(self.depth):
            # Upsample
            x = self.upsample_blocks[i](x)
            
            # Skip connection
            skip_feature = features[self.depth - i - 1]
            
            # Handle cases where dimensions don't match exactly
            if x.shape[2:] != skip_feature.shape[2:]:
                x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
                
            # Concatenate for skip connection
            x = torch.cat((x, skip_feature), dim=1)
            
            # Decode
            x = self.decoder_blocks[i](x)
            
        return x

class MultiChannelUNet(nn.Module):
    """U-Net architecture for multi-channel audio source separation."""
    def __init__(self, input_channels=16, output_channels=16, base_channels=32, 
                 depth=3, dropout_rate=0.2, use_batch_norm=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder
        self.encoder = MultiChannelEncoder(
            input_channels=input_channels,
            base_channels=base_channels,
            depth=depth,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Bottleneck
        bottleneck_channels = base_channels * (2**(depth-1))
        self.bottleneck = MultiChannelConvBlock(
            bottleneck_channels, 
            bottleneck_channels * 2,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Decoder
        self.decoder = MultiChannelDecoder(
            base_channels=base_channels,
            depth=depth,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Final output layer (mask for each channel)
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the multi-channel U-Net.
        
        Args:
            x: Input tensor of shape [batch, channels, freq_bins, time_frames]
                where channels is the number of audio channels
                
        Returns:
            Predicted mask of same shape as input
        """
        # Encoder
        features, bottleneck = self.encoder(x)
        
        # Bottleneck
        bottleneck = self.bottleneck(bottleneck)
        
        # Decoder with skip connections
        decoded = self.decoder(features, bottleneck)
        
        # Generate mask - apply sigmoid for values in [0,1] range
        mask = torch.sigmoid(self.final_conv(decoded))
        
        return mask
