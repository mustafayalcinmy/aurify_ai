# app/generation/generator.py dosyasını güncelle veya bu fonksiyonu ekle

from datetime import datetime
import torch
import os
import logging
import numpy as np # NumPy'ı import etmeyi unutma
from tqdm import tqdm # İsteğe bağlı: Üretim için de progress bar
import time
import torch.nn as nn
# Mevcut generate_music fonksiyonu (sıralı modeller için) burada olabilir
# from ... import ... # Gerekli diğer importlar

# GAN Generator modelini ve MIDI'ye dönüştürme fonksiyonunu import et
# Bu import yollarının projenize göre doğru olduğundan emin olun
try:
    from app.models.gan_generator import Generator
except ImportError:
    Generator = None # Opsiyonel: Eğer model henüz yoksa hata vermesin
try:
    from app.utils.audio_converter import pianoroll_tensor_to_midi
except ImportError:
    pianoroll_tensor_to_midi = None # Opsiyonel



logger = logging.getLogger("MusicGenGenerator") # Logger adını kontrol et

def generate_music(model, model_type, start_sequence, length=100, device='cpu', temperature=1.0):
    model.to(device)
    model.eval()
    generated = start_sequence.copy()
    input_seq = torch.tensor(start_sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad(), tqdm(total=length, desc="Müzik Üretiliyor", unit="nota") as pbar:
        hidden = None  # LSTM/GRU için başlangıç gizli durumu
        
        for _ in range(length):
            # Model tipine göre ileri geçiş
            if model_type in ['lstm', 'gru']:
                output, hidden = model(input_seq, hidden)
                next_token_logits = output[0, -1, :]
            else:
                if model_type == 'transformer':
                    output = model(input_seq)
                else:  # gpt
                    output = model(input_seq)
                next_token_logits = output[0, -1, :]
            
            # Sıcaklık uygulaması (temperature sampling)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            next_token_prob = torch.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(next_token_prob, 1).item()
            generated.append(next_token)
            
            # Sonraki girdi sekansını hazırla
            if model_type in ['lstm', 'gru']:
                input_seq = torch.tensor([[next_token]], dtype=torch.long, device=device)
            else:
                input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
                # Uzunluğu sınırla (bellek tasarrufu için)
                input_seq = input_seq[:, -len(start_sequence):]
            
            pbar.update(1)
            pbar.set_postfix({'son_üretilen': next_token})
    
    return generated

# --- YENİ: GAN Model Üretim Fonksiyonu ---
@torch.no_grad() # Üretim sırasında gradyan hesaplamaya gerek yok
def generate_music_gan(
    generator: nn.Module,      # Eğitilmiş Generator modeli
    latent_dim: int,           # Gürültü vektörü boyutu (constants'dan gelir)
    num_samples: int = 1,      # Üretilecek örnek sayısı (config'den gelir)
    device: str = 'cpu',       # Kullanılacak cihaz ('cpu', 'cuda', 'mps')
    output_dir: str = 'generated_gan', # Çıktı klasörü (config'den gelir)
    filename_pattern: str = "gan_{index}_{timestamp}.mid",
    # pianoroll_tensor_to_midi fonksiyonuna iletilecek parametreler:
    fs: int = 4,               # Zaman çözünürlüğü (constants'dan gelir)
    pitch_range: tuple = (24, 108), # Nota aralığı (constants'dan gelir)
    velocity_threshold: float = 0.1, # Nota aktivasyon eşiği (config'den gelir)
    instrument_program: int = 0,   # Enstrüman program no (config'den gelir)
    bpm: int = 120,            # Tempo (config'den gelir)
    note_duration_steps: int = 1 # Minimum nota süresi (adım cinsinden) - isteğe bağlı
    ):
    """
    Eğitilmiş GAN Generator ile müzik (piyano rulosu) üretir ve MIDI'ye çevirir.

    Args:
        generator (nn.Module): Eğitilmiş Generator modeli.
        latent_dim (int): Gürültü vektörü boyutu.
        num_samples (int): Üretilecek örnek sayısı.
        device (str): Kullanılacak cihaz.
        output_dir (str): Üretilen dosyaların kaydedileceği klasör.
        filename_pattern (str): Kaydedilecek MIDI dosyasının adlandırma deseni.
                                 '{index}' yerine örnek numarası (1'den başlar),
                                 '{timestamp}' yerine zaman damgası (isteğe bağlı) gelir.
        fs (int): Piyano rulosu zaman çözünürlüğü.
        pitch_range (tuple): Kullanılan MIDI nota aralığı (min_pitch, max_pitch).
        velocity_threshold (float): Bir notanın aktif kabul edilmesi için minimum değer (0-1).
        instrument_program (int): Kullanılacak enstrüman program numarası.
        bpm (int): Tempo.
        note_duration_steps (int): Minimum nota süresi (zaman adımı cinsinden).

    Returns:
        list: Başarıyla oluşturulan MIDI dosyalarının yollarının listesi.
    """
    if Generator is None or pianoroll_tensor_to_midi is None:
        logger.error("GAN Generator modeli veya pianoroll_tensor_to_midi fonksiyonu import edilemedi.")
        return []

    if not isinstance(generator, nn.Module):
         logger.error("Geçersiz Generator modeli sağlandı.")
         return []

    generator.to(device)
    generator.eval() # Modeli evaluation moduna al (Dropout, BatchNorm vs. etkilenmez)

    os.makedirs(output_dir, exist_ok=True)
    generated_midi_paths = []

    logger.info(f"{num_samples} adet GAN örneği üretiliyor (Latent Dim: {latent_dim})...")

    for i in tqdm(range(num_samples), desc="GAN Müzik Üretiliyor"):
        try:
            # 1. Rastgele gürültü vektörü oluştur
            noise = torch.randn(1, latent_dim, 1, 1, device=device)

            # 2. Generator ile piyano rulosu tensörünü üret
            fake_pianoroll_tensor = generator(noise)

            # 3. Tensörü CPU'ya alıp NumPy dizisine çevir
            fake_pianoroll_np = fake_pianoroll_tensor.squeeze(0).squeeze(0).cpu().numpy()

            # İsteğe bağlı post-processing (aynı kalır)
            # fake_pianoroll_np = (fake_pianoroll_np + 1.0) / 2.0

            # --- 4. Piyano rulosunu MIDI'ye çevir (Güncellenmiş İsimlendirme) ---
            # Dosya adını desenden oluştur
            now = datetime.now()
            # İstenen format: SaatDakikaSaniye_Gün_Ay_Yıl
            current_time_str = now.strftime("%H%M%S_%d_%m_%Y")
            output_midi_filename = filename_pattern.format(index=i+1, timestamp=current_time_str)
            output_midi_path = os.path.join(output_dir, output_midi_filename)

            midi_path = pianoroll_tensor_to_midi(
                piano_roll=fake_pianoroll_np,
                file_path=output_midi_path, # Oluşturulan yolu kullan
                fs=fs,
                pitch_range=pitch_range,
                velocity_threshold=velocity_threshold,
                note_duration_steps=note_duration_steps,
                instrument_program=instrument_program,
                bpm=bpm
            )

            if midi_path:
                generated_midi_paths.append(midi_path)
            else:
                logger.warning(f"Örnek {i+1} için piyano rulosu MIDI'ye dönüştürülemedi.")

        except Exception as e:
            logger.error(f"GAN örneği {i+1} üretilirken hata oluştu: {e}", exc_info=True)

    logger.info(f"Toplam {len(generated_midi_paths)} adet GAN MIDI dosyası üretildi.")
    return generated_midi_paths