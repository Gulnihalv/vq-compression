import torch
import torch.nn as nn

class VQCompressionModel(nn.Module):
    def __init__(self, encoder, vq, decoder, entropy_model):
        super().__init__()
        self.encoder = encoder
        
        # Encoder'ın output_channels özelliğini güvenli şekilde al
        encoder_output_dim = getattr(encoder, '_output_channels', encoder.output_channels)
        
        self.pre_vq = nn.Conv2d(encoder_output_dim, 
                               vq.embedding_dim, kernel_size=1)
        self.vq = vq
        self.post_vq = nn.Conv2d(vq.embedding_dim, vq.embedding_dim, kernel_size=1)
        self.decoder = decoder
        self.entropy_model = entropy_model
        
        # Store spatial dimensions after encoding for decode method
        self._spatial_shape_cache = {}
        
    def forward(self, x):
        # Store input shape for potential use in decode
        batch_size, _, height, width = x.shape
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        
        # Encoding
        z, features = self.encoder(x)
        z = self.pre_vq(z)
        
        # Vector Quantization
        z_q, indices, vq_loss = self.vq(z)
        
        # Post-processing
        z_q = self.post_vq(z_q)
        
        # Decoding
        recon = self.decoder(z_q, features)
        
        # Entropy model
        logits = self.entropy_model(indices)
        
        return recon, indices, logits, vq_loss
    
    def encode(self, x):
        """Encode input to indices"""
        batch_size, _, height, width = x.shape
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        
        # Store spatial info for decode method
        self._spatial_shape_cache[batch_size] = (encoded_height, encoded_width)
        
        z, _ = self.encoder(x)
        z = self.pre_vq(z)
        _, indices, _ = self.vq(z)
        return indices
    
    def decode(self, indices, spatial_shape=None):
        """
        Decode indices back to image
        Args:
            indices: Quantized indices tensor (B, N) where N = H*W
            spatial_shape: Optional tuple (H, W) for spatial dimensions
                          If None, assumes square images
        """
        batch_size = indices.size(0)
        spatial_size = indices.size(1)
        
        if spatial_shape is not None:
            h, w = spatial_shape
            if h * w != spatial_size:
                raise ValueError(f"Spatial shape {spatial_shape} doesn't match indices size {spatial_size}")
        else:
            # Try to get from cache first
            if batch_size in self._spatial_shape_cache:
                h, w = self._spatial_shape_cache[batch_size]
            else:
                # Square assumption as fallback
                h = w = int(spatial_size ** 0.5)
                if h * w != spatial_size:
                    raise ValueError(f"Cannot infer spatial shape from indices size {spatial_size}. "
                                   f"Please provide spatial_shape parameter.")
        
        # Convert indices to quantized representations
        z_q = self.vq.lookup_indices(indices, (batch_size, h, w))
        z_q = self.post_vq(z_q)
        
        # Decode without encoder features (since we don't have them in decode-only mode)
        recon = self.decoder(z_q)
        return recon
    
    def get_compression_ratio(self, input_shape):
        """
        Calculate theoretical compression ratio
        Args:
            input_shape: Input image shape (B, C, H, W)
        Returns:
            float: Compression ratio (original_bits / compressed_bits)
        """
        batch_size, channels, height, width = input_shape
        
        # Original bits (assuming 8-bit per channel)
        original_bits = channels * height * width * 8
        
        # Compressed bits (log2 of codebook size per spatial location)
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        compressed_bits = encoded_height * encoded_width * torch.log2(torch.tensor(self.vq.num_embeddings)).item()
        
        return original_bits / compressed_bits