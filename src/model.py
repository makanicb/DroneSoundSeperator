# src/multi_channel_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import stft, istft  # Use provided multi-channel STFT functions

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
        for i in range(depth):
            # Calculate in_channels for upsampling block
            if i == 0:
                # Bottleneck output is base_channels * (2 ** depth)
                in_channels = base_channels * (2 ** depth)
            else:
                # Previous decoder block's output is base_channels * (2 ** (depth - i))
                in_channels = base_channels * (2 ** (depth - i))
            
            # Output channels after upsampling
            out_channels = base_channels * (2 ** (depth - i - 1))
            
            # Upsampling block
            self.upsample_blocks.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            
            # Skip feature channels from encoder
            skip_feature_channels = base_channels * (2 ** (depth - i - 1))
            # Decoder block's input is upsampled + skip feature
            decoder_in = out_channels + skip_feature_channels
            
            # Decoder block
            self.decoder_blocks.append(
                MultiChannelConvBlock(
                    decoder_in,
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
            
            # Get corresponding skip feature from encoder
            skip_idx = self.depth - i - 1
            skip_feature = features[skip_idx]
            
            # Adjust spatial dimensions if necessary
            if x.shape[2:] != skip_feature.shape[2:]:
                x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate with skip feature
            x = torch.cat((x, skip_feature), dim=1)
            
            # Apply decoder block
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

class UNetSeparator(nn.Module):
    """
    End-to-end U-Net audio separator that:
    1. Converts raw waveform to complex STFT spectrograms
    2. Predicts a magnitude mask using U-Net
    3. Reconstructs waveform using original phase
    """
    def __init__(
        self, 
        n_fft: int = 2048, 
        hop_length: int = 441, 
        win_length: int = 2048,
        input_channels: int = 16, 
        **unet_kwargs
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # U-Net processes magnitude spectrograms
        self.unet = MultiChannelUNet(
            input_channels=input_channels,
            output_channels=input_channels,  # One mask per channel
            **unet_kwargs
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Input mixture [B, C, T]
        Returns:
            est_waveform: Estimated clean [B, C, T]
        """
        # 1. Compute complex STFT
        complex_spec = stft(waveform, self.n_fft, self.hop_length, self.win_length)
        
        # 2. Get magnitude and phase
        mag_spec = torch.abs(complex_spec)
        phase_spec = torch.angle(complex_spec)
        
        # 3. Predict magnitude mask
        mask = torch.sigmoid(self.unet(mag_spec))  # [B, C, F, T]
        
        # 4. Apply mask to magnitude
        est_mag = mag_spec * mask
        
        # 5. Reconstruct complex spectrogram
        est_complex_spec = est_mag * torch.exp(1j * phase_spec)
        
        # 6. Convert back to waveform
        est_waveform = istft(
            est_complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=waveform.shape[-1]
        )
        return est_waveform
