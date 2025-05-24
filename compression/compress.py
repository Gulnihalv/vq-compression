import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from .arithmetic_coding import ArithmeticCoder

class ImageCompressor:
    """
    Eğitilmiş VQ-VAE modelini kullanarak görüntü sıkıştırma yöntemlerini uygular
    """
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Görüntü sıkıştırıcıyı başlatın
        
        Args:
            model: Eğitilmiş VQCompressionModel
            device: İşlem yapılacak cihaz ('cuda' veya 'cpu')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.coder = ArithmeticCoder(precision=32)
        
    def compress_image(self, image, output_path=None):
        """
        Bir görüntüyü sıkıştırır ve bitstream olarak kaydeder
        
        Args:
            image (torch.Tensor): Sıkıştırılacak görüntü [C, H, W] veya [B, C, H, W]
            output_path (str, optional): Sıkıştırılmış dosyanın kaydedileceği yol (.bin)
            
        Returns:
            dict: Sıkıştırma bilgileri (boyut, oranlar, vb.)
        """
        with torch.no_grad():
            # Batch boyutu ekle (tek görüntü için)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            original_shape = image.shape  # [B, C, H, W]
            image = image.to(self.device)

            # Encoder ile latent kodu çıkar
            z, _ = self.model.encoder(image)
            # 1x1 conv ile embedding_dim'e dönüştürn
            z = self.model.pre_vq(z)
            # VQ aşaması
            z_q, indices, vq_loss = self.model.vq(z)

            # Entropi modelinden olasılıkları al
            logits = self.model.entropy_model(indices)
            probs = F.softmax(logits, dim=-1)

            # İndeks ve olasılıkları düzleştir
            indices_flat = indices[0].cpu()
            probs_list = [probs[0, i].cpu().numpy() for i in range(indices_flat.numel())]

            # (Fallback) Ham indeksleri uint16 olarak byte’lara dök
            arr = indices_flat.numpy().astype(np.uint16)
            bitstream = arr.tobytes()

            # Meta bilgileri hazırla
            latent_shape = (z.shape[2], z.shape[3])
            metadata = {
                "original_shape": [int(x) for x in original_shape[1:]],  # C, H, W orjinal boyut
                "latent_shape": [int(x) for x in latent_shape],         # H', W'
                "num_symbols": int(indices_flat.numel()),
                "dtype": "uint16"
            }

            # Çıktı dosyası belirtilmişse header + metadata + bitstream olarak kaydet
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Metadata JSON
                meta_bytes = json.dumps(metadata).encode('utf-8')
                header_size = len(meta_bytes)
                with open(output_path, 'wb') as f:
                    # Header size (4 byte)
                    f.write(header_size.to_bytes(4, byteorder='big'))
                    # Metadata
                    f.write(meta_bytes)
                    # Bitstream
                    f.write(bitstream)

            # Ölçümleri hesapla
            original_size = np.prod(original_shape[1:]) * 8
            compressed_size = len(bitstream) * 8
            compression_ratio = original_size / compressed_size
            bpp = compressed_size / (original_shape[2] * original_shape[3])

            print(f"[DEBUG] num_symbols={metadata['num_symbols']}, bitstream_bytes={len(bitstream)}")
            return {
                "original_size_bits": original_size,
                "compressed_size_bits": compressed_size,
                "compression_ratio": compression_ratio,
                "bpp": bpp,
                "metadata": metadata,
                "bitstream": bitstream
            }
