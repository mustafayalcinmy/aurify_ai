# app/models/constants_gan.py

# GAN Genel Parametreleri
GAN_TRAINING_PARAMS = {
    "latent_dim": 100,
    "lr_g": 0.0002,
    "lr_d": 0.0002,
    "beta1": 0.5,
    # epochs, batch_size gibi parametreler config'den gelebilir
}

# Piyano Rulosu Veri İşleme Parametreleri (Modele Girdi/Çıktı için önemli)
PIANOROLL_PARAMS = {
    "fs": 4,             # Saniyedeki zaman adımı
    "pitch_min": 24,     # C1
    "pitch_max": 108,    # C8 (84 nota)
    "seq_length": 128,   # GAN'a verilecek zaman adımı sayısı
    "step_size": 64,     # Sliding window adımı
    "include_velocity": True,
}
# Hesaplanan değer
PIANOROLL_PARAMS["num_pitches"] = PIANOROLL_PARAMS["pitch_max"] - PIANOROLL_PARAMS["pitch_min"]

# Generator Modeli Parametreleri (DCGAN örneği üzerinden)
GENERATOR_PARAMS = {
    "output_channels": 1,
    # seq_length ve num_pitches PIANOROLL_PARAMS'dan alınacak
}

# Discriminator Modeli Parametreleri (DCGAN örneği üzerinden)
DISCRIMINATOR_PARAMS = {
    "input_channels": 1,
     # seq_length ve num_pitches PIANOROLL_PARAMS'dan alınacak
}

# GAN Üretim Parametreleri (Varsayılanlar)
GAN_GENERATION_DEFAULTS = {
     "num_samples": 5,
     "velocity_threshold": 0.1,
}