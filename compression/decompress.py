import torch
import numpy as np
import json
import torch.nn.functional as F
from .arithmetic_coding import ArithmeticCoder

class ImageDecompressor:
    """
    Sıkıştırılmış bit akışından görüntüleri geri çözmek için yöntemler uygular
    """
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.coder = ArithmeticCoder(precision=32)

    def decompress_image(self, compressed_file):
        # 1) Dosyayı oku ve header'ı ayıkla
        with open(compressed_file, 'rb') as f:
            data = f.read()
        header_size = int.from_bytes(data[:4], byteorder='big')
        header_data = data[4:4 + header_size]
        bitstream = data[4 + header_size:]

        # 2) Metadata'yı JSON olarak çöz
        metadata = json.loads(header_data.decode('utf-8'))
        # metadata['original_shape'] = [C, H, W]
        _, orig_h, orig_w = metadata['original_shape']
        # metadata['latent_shape'] = [h', w']
        latent_h, latent_w = metadata['latent_shape']
        num_symbols = metadata['num_symbols']
        dtype = metadata.get('dtype', 'uint16')

        # 3) Bitstream'ten ham indeksleri geri oku
        np_dtype = np.uint16 if dtype == 'uint16' else np.int32
        decoded = np.frombuffer(bitstream, dtype=np_dtype, count=num_symbols)

        # 4) PyTorch tensörüne dönüştür ve doğru shape ver
        indices = torch.tensor(decoded, dtype=torch.long, device=self.device).view(1, -1)

        # 5) Model ile geri çözümle: latent_shape'i kesin verelim
        with torch.no_grad():
            recon = self.model.decode(indices, spatial_shape=(latent_h, latent_w))

                    # 6) Orijinal boyuta kesinlikle kırpma veya interpolate etme
        # recon shape: [1, C, pad_h, pad_w]
        recon_h, recon_w = recon.shape[2], recon.shape[3]
        # Crop to at most original
        recon = recon[:, :, :orig_h, :orig_w]
        # Eğer recon boyutu orijinalden küçükse, pad ekle
        pad_h = max(orig_h - recon_h, 0)
        pad_w = max(orig_w - recon_w, 0)
        if pad_h > 0 or pad_w > 0:
            # padding: (left, right, top, bottom)
            recon = F.pad(recon, (0, pad_w, 0, pad_h), mode='reflect')

        return recon
