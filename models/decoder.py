import torch.nn as nn
import torch.nn.functional as F
from utils.helper_models import ResidualBlock, SelfAttention

class Decoder(nn.Module):
    """
    Memory optimized decoder with only final attention
    """
    def __init__(self, out_channels=3, latent_channels=128, num_layers=3, 
                 post_vq_layers=0, downsampling_rate=8, use_attention=True):
        super().__init__()
        
        # Calculate number of upsampling operations needed
        if downsampling_rate in [8, 16]:
            us_num_layers = 3 if downsampling_rate == 8 else 4
        else:
            raise ValueError(f"Downsampling rate {downsampling_rate} not supported. Use 8 or 16.")
            
        self.downsampling_rate = downsampling_rate
        self.use_attention = use_attention
        
        # Post VQ processing layers (without intermediate attention)
        self.post_vq = nn.ModuleList()
        
        for _ in range(post_vq_layers):
            self.post_vq.append(
                nn.Sequential(
                    ResidualBlock(latent_channels),
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(True)
                )
            )
        
        # Upsampling layers with progressive channel reduction
        self.upsample_layers = nn.ModuleList()
        
        # Create channel dimensions for each layer
        channels = [latent_channels]
        for i in range(us_num_layers):
            next_ch = max(latent_channels // (2 ** (i+1)), 32)
            channels.append(next_ch)
        
        # Create upsampling blocks (without intermediate attention)
        for i in range(us_num_layers):
            in_ch = channels[i]
            out_ch = channels[i+1]
            
            up_block = nn.Sequential(
                ResidualBlock(in_ch),
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                ResidualBlock(out_ch)
            )
            
            self.upsample_layers.append(up_block)
        
        # Additional processing layers without upsampling
        num_extra_layers = max(0, num_layers - us_num_layers)
        self.extra_layers = nn.ModuleList()
        
        for i in range(num_extra_layers):
            in_ch = channels[-1] if i == 0 else channels[-1] // 2
            out_ch = channels[-1] // 2
            
            self.extra_layers.append(
                nn.Sequential(
                    ResidualBlock(in_ch),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(True)
                )
            )
            
            # Update the last channel dimension
            if i == num_extra_layers - 1:
                channels.append(out_ch)
        
        # Only final attention before output (if enabled)
        final_channels = channels[-1]
        if use_attention:
            self.final_attention = SelfAttention(final_channels)
        else:
            self.final_attention = None
            
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(final_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1]
        )
        
    def forward(self, z, encoder_features=None):
        # Apply post-VQ processing (no intermediate attention)
        for layer in self.post_vq:
            z = layer(z)
        
        # Apply upsampling layers (no intermediate attention)
        for idx, layer in enumerate(self.upsample_layers):
            z = layer(z)

            # Optional skip connections from encoder features
            if encoder_features is not None and idx < len(encoder_features):
                # Get corresponding encoder feature (in reverse order)
                enc_feat = encoder_features[-(idx+1)]
                
                # Apply skip connection if shapes match
                if z.shape[2:] == enc_feat.shape[2:] and z.shape[1] == enc_feat.shape[1]:
                    z = z + enc_feat
        
        # Apply extra processing layers (no intermediate attention)
        for layer in self.extra_layers:
            z = layer(z)
        
        # Apply final attention only
        if self.final_attention is not None:
            z = self.final_attention(z)
        
        # Apply final output layer
        out = self.final(z)
        return out