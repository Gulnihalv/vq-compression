import torch.nn as nn
from utils.helper_models import ResidualBlock, SelfAttention

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=128, num_layers=3, downsampling_rate=8, use_attention=True):
        super().__init__()
        
        # Calculate number of downsampling operations needed
        if downsampling_rate in [8, 16]:
            ds_num_layers = 3 if downsampling_rate == 8 else 4
        else:
            raise ValueError(f"Downsampling rate {downsampling_rate} not supported. Use 8 or 16.")
            
        self.downsampling_rate = downsampling_rate
        self.use_attention = use_attention
        
        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_channels//4),
            nn.ReLU(True)
        )
        
        # Downsampling layers with progressive channel expansion
        self.downsample_layers = nn.ModuleList()
        
        # Sadece son iki downsampling sonrası attention kullanacağız (belleği korumak için)
        self.attention_modules = nn.ModuleList()
        
        # Channel dimensions for each layer
        channels = [latent_channels//4]
        for i in range(ds_num_layers):
            next_ch = min(channels[-1] * 2, latent_channels)
            channels.append(next_ch)
        
        # Create downsampling blocks
        for i in range(ds_num_layers):
            in_ch = channels[i]
            out_ch = channels[i+1]
            
            down_block = nn.Sequential(
                ResidualBlock(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                ResidualBlock(out_ch)
            )
            
            self.downsample_layers.append(down_block)
            
            # Sadece son iki katmana attention ekle
            if use_attention and i >= ds_num_layers - 2:
                self.attention_modules.append(SelfAttention(out_ch))
        
        # Additional processing layers without downsampling
        num_extra_layers = max(0, num_layers - ds_num_layers)
        self.extra_layers = nn.ModuleList()
        
        # Extra katmanlarda da seçici olarak attention kullan
        if num_extra_layers > 0 and use_attention:
            self.extra_attention = SelfAttention(channels[-1])
        else:
            self.extra_attention = None
        
        for _ in range(num_extra_layers):
            self.extra_layers.append(
                nn.Sequential(
                    ResidualBlock(channels[-1]),
                    nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[-1]),
                    nn.ReLU(True)
                )
            )
        
        # Final attention sadece gerekliyse kullan
        if use_attention:
            self.final_attention = SelfAttention(channels[-1])
        else:
            self.final_attention = None
        
        # Store final output channels for easy access
        self._output_channels = channels[-1]
            
    def forward(self, x):
        x = self.initial(x)
        
        # Store intermediate features for possible skip connections
        features = [x]
        
        # Apply downsampling layers with selective attention
        for i, layer in enumerate(self.downsample_layers):
            x = layer(x)
            
            # Burası değiştirildi
            if self.use_attention and i >= len(self.downsample_layers) - 2:
                attention_idx = i - (len(self.downsample_layers) - 2)
                if attention_idx < len(self.attention_modules):
                    x = self.attention_modules[attention_idx](x)

            features.append(x)
        
        # Apply extra processing layers
        for layer in self.extra_layers:
            x = layer(x)
            features.append(x)
        
        # Sadece tek bir final attention uygula (eğer aktifse)
        if self.final_attention is not None:
            x = self.final_attention(x)
            
        return x, features
    
    @property
    def output_channels(self):
        """Returns the number of output channels from the encoder"""
        return self._output_channels
    
    def get_output_shape(self, input_shape):
        """
        Calculate the output shape given an input shape
        Args:
            input_shape: tuple (C, H, W) or (B, C, H, W)
        Returns:
            tuple: Output shape
        """
        if len(input_shape) == 4:
            batch_size, channels, height, width = input_shape
        else:
            channels, height, width = input_shape
            batch_size = None
        
        # Calculate spatial dimensions after downsampling
        output_height = height // self.downsampling_rate
        output_width = width // self.downsampling_rate
        
        if batch_size is not None:
            return (batch_size, self.output_channels, output_height, output_width)
        else:
            return (self.output_channels, output_height, output_width)