import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    
    This implementation uses EMA updates for the codebook instead of using
    backpropagation, which can lead to improved codebook usage.
    """
    def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_coef=0.25, decay=0.99):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_coef = commitment_coef
        self.decay = decay
        
        # Initialize codebook as a nn.Embedding layer
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embed.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA related buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embed.weight.data.clone())
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        
    def forward(self, z_e):
        """
        Quantize the input tensor and compute the quantization loss.
        
        Args:
            z_e (Tensor): Input tensor [B, D, H, W]
            
        Returns:
            tuple: (quantized tensor, indices, loss)
        """
        # Reshape input for distance calculation
        input_shape = z_e.shape
        flat_input = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embed.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embed.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = self.embed(encoding_indices).view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # EMA update (only during training)
        if self.training:
            # Use EMA to update the embedding vectors
            self._ema_update(encodings, flat_input)
        
        # Compute loss
        # Commitment loss: encoder should commit to its outputs
        commitment_loss = F.mse_loss(quantized.detach(), z_e)
        
        # Make gradient flow back to encoder from decoder
        quantized = z_e + (quantized - z_e).detach()  # Straight-through estimator
        
        # Reshape indices for batch format
        indices = encoding_indices.view(input_shape[0], input_shape[2] * input_shape[3])
        
        return quantized, indices, commitment_loss * self.commitment_coef
    
    def _ema_update(self, encodings, flat_input):
        """
        Update the codebook using Exponential Moving Average
        
        Args:
            encodings: One-hot encoding of indices
            flat_input: Flattened input tensor
        """
        # EMA update for embedding vectors
        self.cluster_size = encodings.sum(0)
        
        # Laplace smoothing of the cluster size
        n = self.cluster_size.sum()
        
        # Update ema_cluster_size
        self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * self.cluster_size
        
        # Compute embeddings from the current batch
        dw = torch.matmul(encodings.t(), flat_input)
        
        # Update ema_w
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        
        # Normalize cluster sizes
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)
        
        # Update embeddings
        normalized_ema_w = self.ema_w / cluster_size.unsqueeze(1)
        self.embed.weight.data.copy_(normalized_ema_w)
        
    def quantize(self, z_e):
        """
        Quantize the input tensor without computing loss.
        
        Args:
            z_e (Tensor): Input tensor [B, D, H, W]
            
        Returns:
            tuple: (quantized tensor, indices)
        """
        # Reshape input
        input_shape = z_e.shape
        flat_input = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embed.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embed.weight.t()))
        
        # Get indices of nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized = self.embed(encoding_indices).view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices
        indices = encoding_indices.view(input_shape[0], input_shape[2] * input_shape[3])
        
        return quantized, indices
    
    def lookup_indices(self, indices, spatial_shape): # burası son güncelleme ile değiştirildi
        """
        Convert indices back to their corresponding embeddings.
        """
        batch_size, h, w = spatial_shape

        # Validate input
        expected_spatial_size = h * w
        actual_spatial_size = indices.size(1)
        if expected_spatial_size != actual_spatial_size:
            raise ValueError(f"Spatial shape mismatch: expected {expected_spatial_size}, got {actual_spatial_size}")

        # Clamp indices to prevent out-of-bounds errors (EKLENEN SATIR)
        indices = torch.clamp(indices, 0, self.num_embeddings - 1)

        # Flatten indices and lookup embeddings
        flat_indices = indices.view(-1)
        embeddings = self.embed(flat_indices)

        # Reshape to spatial format
        z_q = embeddings.view(batch_size, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]

        return z_q
    
    def get_codebook_usage(self):
        """
        Get statistics about codebook usage
        
        Returns:
            dict: Statistics including usage ratio, entropy, etc.
        """
        with torch.no_grad():
            total_size = self.ema_cluster_size.sum()
            if total_size > 0:
                # Calculate usage ratio (non-zero entries)
                used_codes = (self.ema_cluster_size > 1e-7).sum().item()
                usage_ratio = used_codes / self.num_embeddings
                
                # Calculate entropy of cluster distribution
                probs = self.ema_cluster_size / total_size
                probs = probs[probs > 1e-7]  # Remove zeros for log calculation
                entropy = -torch.sum(probs * torch.log(probs)).item()
                max_entropy = torch.log(torch.tensor(self.num_embeddings)).item()
                normalized_entropy = entropy / max_entropy
                
                return {
                    'usage_ratio': usage_ratio,
                    'used_codes': used_codes,
                    'total_codes': self.num_embeddings,
                    'entropy': entropy,
                    'normalized_entropy': normalized_entropy,
                    'cluster_sizes': self.ema_cluster_size.cpu().numpy()
                }
            else:
                return {
                    'usage_ratio': 0.0,
                    'used_codes': 0,
                    'total_codes': self.num_embeddings,
                    'entropy': 0.0,
                    'normalized_entropy': 0.0,
                    'cluster_sizes': self.ema_cluster_size.cpu().numpy()
                }