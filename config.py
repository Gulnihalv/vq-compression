import torch

# Cihaz yapılandırması
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Veri yolu yapılandırması
TRAIN_DATA_DIR = "data/DIV2K/DIV2K_train_HR"
VAL_DATA_DIR = "data/DIV2K/DIV2K_valid_HR"
TEST_DATA_DIR = "data/test"
MODEL_SAVE_DIR = "checkpoints"

# Model yapılandırması
MODEL_CONFIG = {
    # Encoder parametreleri
    "encoder": {
        "in_channels": 3,
        "latent_channels": 192,  # 128'den artırıldı
        "num_layers": 4,
        "downsampling_rate": 8,
        "use_attention": True
    },
    
    # Vector Quantizer parametreleri
    "vq": {
        "num_embeddings": 2048,  # 1024'ten artırıldı
        "embedding_dim": 192,    # encoder ile uyumlu
        "commitment_coef": 0.25, # 0.1'den artırıldı
        "decay": 0.99           # EMA decay rate
    },
    
    # Decoder parametreleri
    "decoder": {
        "out_channels": 3,
        "latent_channels": 192,  # encoder ile uyumlu
        "num_layers": 4,
        "post_vq_layers": 2,    # VQ sonrası işleme katmanları
        "downsampling_rate": 8,
        "use_attention": True
    },
    
    # Entropi modeli parametreleri
    "entropy_model": {
        "num_embeddings": 2048,  # VQ ile uyumlu
        "num_layers": 4,         # 3'ten artırıldı
        "hidden_dim": 384,       # 256'dan artırıldı
        "num_heads": 8           # 4'ten artırıldı
    }
}

# GELİŞMİŞ LOSS KONFIGÜRASYONU
LOSS_CONFIG = {
    # Ana loss ağırlıkları
    "weights": {
        "l1_weight": 1.0,
        "l2_weight": 0.3,           # Biraz azaltıldı
        "perceptual_weight": 0.2,   # Artırıldı - görsel kalite için kritik
        "ms_ssim_weight": 0.15,     # Azaltıldı ama hala aktif
        "vq_weight": 1.0,
        "entropy_weight": 0.01
    },
    
    # Perceptual loss konfigürasyonu
    "perceptual": {
        "layers": ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1'],
        "layer_weights": [0.25, 0.5, 0.75, 1.0]  # Daha derin katmanlara daha fazla ağırlık
    },
    
    # MS-SSIM konfigürasyonu
    "ms_ssim": {
        "data_range": 1.0,
        "weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    },
    
    # Adaptive loss konfigürasyonu
    "adaptive": {
        "enabled": True,
        "adaptation_rate": 0.005,  # Daha yavaş adaptasyon
        "target_ratios": {
            "l1": 0.3,
            "perceptual": 0.2,
            "ms_ssim": 0.1,
            "vq": 0.3,
            "entropy": 0.1
        }
    },
    
    # Warmup konfigürasyonu
    "warmup": {
        "enabled": True,
        "epochs": 15,  # İlk 15 epoch warmup
        "schedule": {
            "perceptual": "quadratic",  # Perceptual loss için kademeli
            "ms_ssim": "linear"         # MS-SSIM için doğrusal
        }
    }
}

# Eğitim yapılandırması
TRAIN_CONFIG = {
    "batch_size": 20,          # Perceptual loss için biraz düşürüldü (GPU memory)
    "num_workers": 4,
    "learning_rate": 6e-5,     # Daha kararlı eğitim için düşürüldü
    "weight_decay": 1e-5,
    "epochs": 200,             # Daha uzun eğitim - perceptual loss için gerekli
    "save_every": 5,
    "early_stopping_patience": 30,  # Daha uzun sabır
    
    # Loss ağırlıkları (LOSS_CONFIG'e taşındı ama geriye uyumluluk için)
    "reconstruction_weight": 1.0,
    "vq_weight": 1.0,
    "entropy_weight": 0.01,
    "perceptual_weight": 0.2,  # YENİ
    "ms_ssim_weight": 0.15,    # YENİ
    
    "max_train_samples": None,
    "max_val_samples": None,
    
    # Learning rate scheduler
    "scheduler": {
        "type": "cosine_restart",
        "T_0": 40,                 # Daha uzun cycle
        "T_mult": 2,
        "eta_min": 5e-7           # Daha düşük minimum
    },
    
    # Gradient clipping - perceptual loss için kritik
    "gradient_clipping": {
        "enabled": True,
        "max_norm": 1.0
    }
}

# Veri ön işleme yapılandırması
DATA_CONFIG = {
    "image_size": 256,
    "crop_size": 256,
    "horizontal_flip": True,
    "augmentation_level": "progressive",
    
    # Progressive augmentation detayları - perceptual loss ile uyumlu
    "augmentation_schedule": {
        "epochs_0_40": {
            "rotation": 3,          # Daha az agresif
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.05,
            "hue": 0.02
        },
        "epochs_41_120": {
            "rotation": 8,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05,
            "gaussian_blur": 0.1    # Hafif blur
        },
        "epochs_121_200": {
            "rotation": 12,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.08,
            "gaussian_blur": 0.2,
            "random_erasing": 0.05  # Çok hafif erasing
        }
    }
}

TRAINING_STRATEGY = {
    "progressive_augmentation": {
        "enabled": True,
        "schedule": {
            "epochs_0_40": "custom", 
            "epochs_41_120": "custom", 
            "epochs_121_200": "custom"
        }
    }
}

# Test/değerlendirme yapılandırması
EVAL_CONFIG = {
    "batch_size": 12,  # Perceptual loss hesabı için daha düşük
    "metrics": ["psnr", "ms_ssim", "bpp", "lpips"]  # LPIPS eklendi
}

# Sıkıştırma yapılandırması
COMPRESSION_CONFIG = {
    "arithmetic_coder_precision": 32
}

# Optimizasyon yapılandırması
OPTIMIZATION_CONFIG = {
    "warmup_epochs": 15,        # Loss warmup ile uyumlu
    "gradient_clip_norm": 1.0,
    "accumulation_steps": 2,
    
    # Multi-scale training - perceptual loss için faydalı
    "multiscale_training": {
        "enabled": True,
        "scales": [224, 256, 288],  # Daha az agresif
        "change_frequency": 8       # Daha az sık değişim
    },
    
    # Mixed precision training
    "mixed_precision": {
        "enabled": True,
        "opt_level": "O1"  # Daha kararlı
    }
}

# Logging ve monitoring
LOGGING_CONFIG = {
    "log_every": 50,  # Her 50 batch'te log
    "image_log_every": 5,  # Her 5 epoch'ta örnek görüntüler
    "tensorboard": {
        "enabled": True,
        "log_dir": "runs",
        "log_images": True,
        "log_histograms": False  # GPU memory tasarrufu
    },
    "wandb": {
        "enabled": False,  # İsteğe bağlı
        "project_name": "vq-compression",
        "entity": None
    }
}

# Model checkpoint stratejisi
CHECKPOINT_CONFIG = {
    "save_best_only": False,
    "save_last": True,
    "save_top_k": 3,
    "monitor": "val_loss",
    "mode": "min",
    "auto_resume": True
}