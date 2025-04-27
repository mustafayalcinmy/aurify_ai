# app/data/dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import math

# Gerekli fonksiyonları import et
# Bu import yolunun doğru olduğundan emin ol
from app.utils.audio_converter import midi_to_sequence, midi_to_pianoroll_tensor
import pretty_midi # Gerekli olabilir

logger = logging.getLogger("MusicGenDataset") # Logger adını kontrol et


# Veri seti sınıfı: Maestro veri kümesindeki MIDI dosyalarını okur ve sequence'leri oluşturur.
class MusicDataset(Dataset):
    def __init__(self, data_dir, seq_length=50):
        """
        data_dir: MIDI dosyalarının bulunduğu dizin.
        seq_length: Eğitim için kullanılacak ardışık token (nota) sayısı.
        """
        self.seq_length = seq_length
        self.midi_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.mid') or f.endswith('.midi')
        ]
        self.sequences = []
        for midi_file in self.midi_files:
            seq = midi_to_sequence(midi_file)
            if seq is not None and len(seq) > seq_length:
                # Sliding window yöntemi ile ardışık örnekler oluşturuyoruz
                for i in range(len(seq) - seq_length):
                    self.sequences.append(seq[i:i+seq_length+1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Giriş dizisi ve hedef diziyi oluşturuyoruz (bir adım kaydırılmış)
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq
    

class PianoRollDataset(Dataset):
    """MIDI dosyalarını piyano rulosu tensörlerine dönüştürür (GAN için)."""
    def __init__(self, data_dir, seq_length=128, fs=4, pitch_range=(24, 108), include_velocity=True, step_size=None):
        """
        Args:
            data_dir (str): MIDI dosyalarının bulunduğu dizin.
            seq_length (int): GAN'a verilecek zaman adımı sayısı.
            fs (int): Piyano rulosu zaman çözünürlüğü (saniyedeki adım).
            pitch_range (tuple): Kullanılacak MIDI nota aralığı (min_pitch, max_pitch).
            include_velocity (bool): Velocity bilgisini dahil et.
            step_size (int, optional): Sliding window adımı. None ise seq_length kadar atlar.
        """
        self.seq_length = seq_length
        self.pitch_range = pitch_range
        if pitch_range[1] <= pitch_range[0]:
             raise ValueError(f"Geçersiz pitch_range: {pitch_range}")
        self.num_pitches = pitch_range[1] - pitch_range[0]
        self.pianorolls = [] # Piyano rulosu segmentlerini (NumPy dizileri olarak) tutacak liste

        if step_size is None:
            step_size = seq_length # Varsayılan adım boyutu

        # Alt klasörler dahil tüm MIDI dosyalarını bul
        midi_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.endswith(('.mid', '.midi'))
        ]

        logger.info(f"{len(midi_files)} MIDI dosyası (pianoroll) için işleniyor...")
        processed_file_count = 0
        skipped_file_count = 0

        for midi_file in midi_files:
            # Her MIDI dosyasını piyano rulosu tensörüne çevir
            piano_roll = midi_to_pianoroll_tensor(
                midi_path=midi_file,
                fs=fs,
                pitch_range=pitch_range,
                include_velocity=include_velocity
            )

            # Dönüşüm başarılıysa ve yeterli uzunluktaysa segmentlere ayır
            if piano_roll is not None and piano_roll.shape[0] >= seq_length:
                processed_file_count += 1
                # Sliding window yöntemiyle segmentler oluştur
                for i in range(0, piano_roll.shape[0] - seq_length + 1, step_size):
                    roll_segment = piano_roll[i : i + seq_length, :] # Shape: (seq_length, num_pitches)
                    # GAN için genellikle (Kanal, Zaman, Pitch) formatı beklenir. Kanal=1 ekleyelim.
                    # Shape: (1, seq_length, num_pitches)
                    self.pianorolls.append(roll_segment.reshape(1, seq_length, self.num_pitches))
            elif piano_roll is not None:
                # logger.debug(f"Skipping short file: {os.path.basename(midi_file)} (Length: {piano_roll.shape[0]} < {seq_length})")
                skipped_file_count += 1
            else:
                # midi_to_pianoroll_tensor içinde zaten loglama yapılıyor
                skipped_file_count += 1


        if not self.pianorolls:
             logger.warning(f"Hiç uygun piyano rulosu segmenti oluşturulamadı! ({processed_file_count} dosya işlendi, {skipped_file_count} dosya atlandı).")
        else:
             logger.info(f"{processed_file_count} dosyadan {len(self.pianorolls)} adet piyano rulosu segmenti oluşturuldu ({skipped_file_count} dosya atlandı/kısaydı).")


    def __len__(self):
        """Veri setindeki toplam segment sayısını döndürür."""
        return len(self.pianorolls)

    def __getitem__(self, idx):
        """
        Belirtilen indeksteki piyano rulosu segmentini PyTorch tensörü olarak döndürür.
        """
        # NumPy dizisini PyTorch tensörüne çevir
        roll_tensor = torch.tensor(self.pianorolls[idx], dtype=torch.float32)

        # İsteğe bağlı: Normalizasyon (-1, 1 aralığına)
        # Eğer include_velocity=True ise veriler [0, 1] aralığındadır.
        # GAN'lar genellikle [-1, 1] aralığını tercih eder (Generator'da Tanh aktivasyonu ile uyumlu).
        # if roll_tensor.max() <= 1.0 and roll_tensor.min() >= 0.0:
        #     roll_tensor = (roll_tensor * 2.0) - 1.0

        return roll_tensor