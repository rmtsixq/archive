import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block used in U-Net architecture.
    Consists of two 3x3 convolutions, each followed by batch normalization and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for landslide segmentation.
    
    Output: Landslide probability map (0-1 range) where each pixel value represents
    the probability of that pixel being part of a landslide.
    
    The U-Net consists of:
    - Encoder (downsampling path): Extracts features from the input image
    - Decoder (upsampling path): Reconstructs the landslide probability map
    - Skip connections: Connect encoder and decoder to preserve spatial information
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling path)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling path)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch (due to pooling)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        # Final output: Landslide probability map (0-1 range)
        x = self.final_conv(x)
        return torch.sigmoid(x)  # Sigmoid outputs landslide probability (0-1)


def get_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and return a U-Net model instance.
    """
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    return model

