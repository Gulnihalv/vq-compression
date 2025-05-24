import torch
from utils import load_image, save_image
from models import VQCompressionModel
from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizerEMA
from models.entropy import EntropyModel
from config import MODEL_CONFIG, DEVICE
from compression import ImageCompressor, ImageDecompressor

def main():
    # Model yapılandırması
    encoder = Encoder(
        in_channels=MODEL_CONFIG['encoder']['in_channels'],
        latent_channels=MODEL_CONFIG['encoder']['latent_channels'],
        num_layers=MODEL_CONFIG['encoder']['num_layers']
    )
    
    vq = VectorQuantizerEMA(
        num_embeddings=MODEL_CONFIG['vq']['num_embeddings'],
        embedding_dim=MODEL_CONFIG['vq']['embedding_dim'],
        commitment_coef=MODEL_CONFIG['vq']['commitment_coef']
    )
    
    decoder = Decoder(
        out_channels=MODEL_CONFIG['decoder']['out_channels'],
        latent_channels=MODEL_CONFIG['decoder']['latent_channels'],
        num_layers=MODEL_CONFIG['decoder']['num_layers']
    )
    
    entropy_model = EntropyModel(
        num_embeddings=MODEL_CONFIG['entropy_model']['num_embeddings'],
        num_layers=MODEL_CONFIG['entropy_model']['num_layers'],
        hidden_dim=MODEL_CONFIG['entropy_model']['hidden_dim'],
        num_heads=MODEL_CONFIG['entropy_model']['num_heads']
    )
    
    model = VQCompressionModel(
        encoder=encoder,
        vq=vq,
        decoder=decoder,
        entropy_model=entropy_model
    ).to(DEVICE)

    # Checkpoint yükleme
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Sıkıştırma/Açma işlemleri
    input_tensor = load_image("data/DIV2K/DIV2K_train_HR/0002.png")
    compressor = ImageCompressor(model, DEVICE)
    decompressor = ImageDecompressor(model, DEVICE)

    # Sıkıştır ve kaydet
    compressed_info = compressor.compress_image(
        input_tensor, 
        "output/compressed.bin"
    )
    
    # Geri aç ve kaydet
    reconstructed = decompressor.decompress_image("output/compressed.bin")
    save_image(reconstructed, "output/reconstructed.jpg")

    # Metrikler
    print(f"➤ Sıkıştırma Oranı: {compressed_info['compression_ratio']:.2f}x")
    print(f"➤ Bit-per-Pixel: {compressed_info['bpp']:.2f}")
    
    # PSNR Hesapla
    mse = torch.mean((input_tensor.cpu() - reconstructed.cpu())**2)
    psnr_val = 10 * torch.log10(1 / mse)
    print(f"➤ PSNR: {psnr_val.item():.2f} dB")

if __name__ == "__main__":
    main()