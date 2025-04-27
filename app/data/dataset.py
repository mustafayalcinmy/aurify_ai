import os
import torch
from torch.utils.data import Dataset

from utils.audio_converter import midi_to_sequence


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