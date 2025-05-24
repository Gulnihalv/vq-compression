import torch.nn as nn
import torch.nn.functional as F
from utils.helper_models import ResidualBlock, SelfAttention

class Decoder(nn.Module):
    """
    RAM kullanımı optimize edilmiş decoder (seçici attention kullanımı ile)
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
        
        # Post VQ processing layers (optimize edilmiş)
        self.post_vq = nn.ModuleList()
        
        # Sadece tek bir post-VQ attention kullan (bellek optimizasyonu için)
        if post_vq_layers > 0 and use_attention:
            self.post_vq_attention = SelfAttention(latent_channels)
        else:
            self.post_vq_attention = None
        
        for _ in range(post_vq_layers):
            self.post_vq.append(
                nn.Sequential(
                    ResidualBlock(latent_channels),
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(True)
                )
            )
        
        # Initial attention (tek bir attention kullan)
        if use_attention:
            self.initial_attention = SelfAttention(latent_channels)
        else:
            self.initial_attention = None
        
        # Upsampling layers with progressive channel reduction
        self.upsample_layers = nn.ModuleList()
        
        # Sadece stratejik noktalarda attention kullan (bellek optimizasyonu için)
        self.upsample_attention = nn.ModuleList()
        
        # Create channel dimensions for each layer
        channels = [latent_channels]
        for i in range(us_num_layers):
            next_ch = max(latent_channels // (2 ** (i+1)), 32)
            channels.append(next_ch)
        
        # Create upsampling blocks
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
            
            # Sadece son iki upsampling katmanına attention ekle (optimizasyon için)
            if use_attention and i >= us_num_layers - 2:
                self.upsample_attention.append(SelfAttention(out_ch))
        
        # Additional processing layers without upsampling
        num_extra_layers = max(0, num_layers - us_num_layers)
        self.extra_layers = nn.ModuleList()
        
        # Extra katmanlar için tek bir attention modülü (bellek optimizasyonu için)
        if num_extra_layers > 0 and use_attention:
            last_ch = channels[-1] // 2 if num_extra_layers > 0 else channels[-1]
            self.extra_attention = SelfAttention(last_ch)
        else:
            self.extra_attention = None
        
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
            
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(channels[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1]
        )
        
    def forward(self, z, encoder_features=None):
        # Apply post-VQ processing (optimize edilmiş attention kullanımı)
        for idx, layer in enumerate(self.post_vq):
            z = layer(z)
            
            # Sadece son post-VQ işleminden sonra attention uygula
            if idx == len(self.post_vq) - 1 and self.post_vq_attention is not None:
                z = self.post_vq_attention(z)
        
        # Apply initial attention if available
        if self.initial_attention is not None:
            z = self.initial_attention(z)
        
        # Apply upsampling layers with selective attention
        for idx, layer in enumerate(self.upsample_layers):
            z = layer(z)
            
            # Dikkat et: Tüm katmanlarda değil, sadece seçilen katmanlarda attention kullan
            if self.use_attention and idx >= len(self.upsample_layers) - 2:
                attention_idx = idx - (len(self.upsample_layers) - 2)
                if attention_idx < len(self.upsample_attention):
                    z = self.upsample_attention[attention_idx](z)

            # Optional skip connections from encoder features
            if encoder_features is not None and idx < len(encoder_features):
                # Get corresponding encoder feature (in reverse order)
                enc_feat = encoder_features[-(idx+1)]
                
                # Apply skip connection if shapes match
                if z.shape[2:] == enc_feat.shape[2:] and z.shape[1] == enc_feat.shape[1]:
                    z = z + enc_feat
        
        # Apply extra processing layers with selective attention
        for idx, layer in enumerate(self.extra_layers):
            z = layer(z)
            
            # Sadece son extra layer'dan sonra attention uygula
            if idx == len(self.extra_layers) - 1 and self.extra_attention is not None:
                z = self.extra_attention(z)
        
        # Apply final output layer
        out = self.final(z)
        return out
