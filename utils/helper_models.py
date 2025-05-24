import torch
import torch.nn as nn
import torch.nn.functional as F

# attension mekanizması yeni eklendi !
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8, reduction_factor=2, use_positional=True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.reduction_factor = reduction_factor
        self.use_positional = use_positional
        
        # Head dimension kontrolü
        assert channels % num_heads == 0, f"Channels {channels} must be divisible by num_heads {num_heads}"
        self.head_dim = channels // num_heads
        
        # Daha az agresif spatial reduction
        if reduction_factor > 1:
            self.spatial_reduce = nn.Conv2d(channels, channels, 
                                          kernel_size=reduction_factor, 
                                          stride=reduction_factor, 
                                          padding=0, groups=channels//4)
        else:
            self.spatial_reduce = None
            
        # Multi-head projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_drop = nn.Dropout(0.1)
        
        # Learnable scaling parameter (başlangıçta küçük)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Layer normalization
        self.norm = nn.GroupNorm(num_groups=min(32, channels//4), num_channels=channels)
        
        # Positional encoding (opsiyonel)
        if use_positional:
            self.pos_embedding = nn.Parameter(torch.randn(1, channels, 32, 32) * 0.02)
        else:
            self.pos_embedding = None
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalization
        identity = x
        x = self.norm(x)
        
        # Spatial reduction if needed (daha az agresif)
        if self.spatial_reduce is not None and H > 16 and W > 16:
            x_reduced = self.spatial_reduce(x)
            _, _, H_r, W_r = x_reduced.shape
        else:
            x_reduced = x
            H_r, W_r = H, W
            
        # Positional encoding
        if self.pos_embedding is not None:
            pos_emb = F.interpolate(self.pos_embedding, size=(H_r, W_r), mode='bilinear', align_corners=False)
            x_reduced = x_reduced + pos_emb
        
        # Compute Q, K, V
        q = self.query(x_reduced)  # B, C, H_r, W_r
        k = self.key(x_reduced)    # B, C, H_r, W_r  
        v = self.value(x_reduced)  # B, C, H_r, W_r
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H_r * W_r).permute(0, 1, 3, 2)  # B, heads, HW, head_dim
        k = k.view(B, self.num_heads, self.head_dim, H_r * W_r)  # B, heads, head_dim, HW
        v = v.view(B, self.num_heads, self.head_dim, H_r * W_r).permute(0, 1, 3, 2)  # B, heads, HW, head_dim
        
        # Scaled dot-product attention with memory optimization
        scale = self.head_dim ** -0.5
        
        # Chunked attention for memory efficiency
        chunk_size = min(H_r * W_r, 256)  # Process in chunks
        attn_output = torch.zeros_like(v)
        
        for i in range(0, H_r * W_r, chunk_size):
            end_i = min(i + chunk_size, H_r * W_r)
            q_chunk = q[:, :, i:end_i]  # B, heads, chunk, head_dim
            
            # Compute attention for this chunk
            attn_chunk = torch.matmul(q_chunk, k) * scale  # B, heads, chunk, HW
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            
            # Apply attention
            attn_output[:, :, i:end_i] = torch.matmul(attn_chunk, v)
        
        # Reshape back
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()
        attn_output = attn_output.view(B, C, H_r, W_r)
        
        # Interpolate back to original size if needed
        if H_r != H or W_r != W:
            attn_output = F.interpolate(attn_output, size=(H, W), mode='bilinear', align_corners=False)
        
        # Output projection
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        
        # Residual connection with learnable scaling
        return identity + self.gamma * attn_output


class ChannelAttention(nn.Module):
    """
    Channel-wise attention modülü - spatial attention'a alternatif/ek
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class HybridAttention(nn.Module):
    """
    Spatial ve channel attention'ı birleştiren hibrit modül
    """
    def __init__(self, channels, num_heads=8, reduction_factor=2):
        super().__init__()
        self.spatial_attn = SelfAttention(channels, num_heads, reduction_factor)
        self.channel_attn = ChannelAttention(channels)
        
        # Attention türlerini dengelemek için learnable weights
        self.spatial_weight = nn.Parameter(torch.tensor(0.5))
        self.channel_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # İki attention türünü sıralı uygula
        x_spatial = self.spatial_attn(x)
        x_channel = self.channel_attn(x_spatial)
        
        # Weighted combination
        spatial_contrib = self.spatial_weight * (x_spatial - x)
        channel_contrib = self.channel_weight * (x_channel - x_spatial)
        
        return x + spatial_contrib + channel_contrib

class ResidualBlock(nn.Module):
    """
    Residual block for image processing with batch normalization
    """
    def __init__(self, channels, use_attention=True, attention_type="hybrid"):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=min(32, channels//4), num_channels=channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=min(32, channels//4), num_channels=channels)
        
        # Attention modülü
        if use_attention:
            if attention_type == "spatial":
                self.attention = SelfAttention(channels, num_heads=8)
            elif attention_type == "channel":
                self.attention = ChannelAttention(channels)
            elif attention_type == "hybrid":
                self.attention = HybridAttention(channels)
            else:
                self.attention = None
        else:
            self.attention = None
            
        # Activation
        self.activation = nn.GELU()  # ReLU yerine GELU
        
    def forward(self, x):
        identity = x
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Attention uygula
        if self.attention is not None:
            out = self.attention(out)
        
        out += identity
        return self.activation(out)