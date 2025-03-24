import os
import math
import torch
import pretty_midi
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import configparser
import logging
import argparse
import json
from pydub import AudioSegment
import subprocess
import random

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MusicGen")

# MIDI dosyalarını dizi haline getiren yardımcı fonksiyon
def midi_to_sequence(midi_path):
    """
    MIDI dosyasını yükler ve basitçe notaların pitch değerlerini içeren bir liste oluşturur.
    (Not: Daha sofistike zaman/dinamik temsilleri için ek işleme gerekebilir.)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        logger.error(f"MIDI yüklenirken hata: {e}")
        return None

    sequence = []
    # Tüm enstrümanlarda gezinirken sadece piyano (drum olmayan) enstrümanları ele alıyoruz
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            sequence.append(note.pitch)
    return sequence


def sequence_to_midi(sequence, file_path='generated_music.mid', bpm=120, note_duration=0.5, instrument_program=0):
    """
    Pitch dizisini MIDI dosyasına dönüştürür.
    
    Parametreler:
    sequence: MIDI pitch değerlerinin listesi (0-127)
    file_path: Çıktı dosyasının yolu
    bpm: Dakikadaki vuruş sayısı (tempo)
    note_duration: Her notanın saniye cinsinden süresi
    instrument_program: MIDI program numarası (enstrüman türü)
    """
    # MIDI objesi ve enstrüman oluştur
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)
    
    # Zaman hesaplamaları
    tempo = bpm
    seconds_per_beat = 60.0 / tempo
    current_time = 0.0
    
    # Her pitch için nota ekle
    for pitch in sequence:
        # Notayı oluştur (başlangıç zamanı, bitiş zamanı, pitch, velocity)
        note = pretty_midi.Note(
            velocity=100,  # Ses şiddeti (0-127)
            pitch=pitch,
            start=current_time,
            end=current_time + note_duration
        )
        instrument.notes.append(note)
        
        # Sonraki nota için zamanı güncelle
        current_time += note_duration * 0.5  # Notaların %50 örtüşmesi için
    
    # Enstrümanı MIDI'ye ekle ve dosyayı kaydet
    midi.instruments.append(instrument)
    midi.write(file_path)
    return file_path


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

# LSTM tabanlı müzik üretim modeli
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=64, hidden_dim=128, num_layers=1):
        """
        vocab_size: MIDI pitch değerleri (0-127) için kelime dağarcığı boyutu.
        embedding_dim: Gömme (embedding) boyutu.
        hidden_dim: LSTM gizli katman boyutu.
        num_layers: LSTM katman sayısı.
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x boyutu: (batch_size, seq_length)
        embed = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)   # (batch_size, seq_length, vocab_size)
        return output, hidden

# Positional Encoding: Transformer modelinde sıralı bilgiyi eklemek için
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: Gömme boyutu.
        dropout: Dropout oranı.
        max_len: Maksimum dizi uzunluğu.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (seq_length, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer tabanlı müzik üretim modeli
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        vocab_size: MIDI pitch dağarcığı boyutu.
        embedding_dim: Gömme boyutu.
        num_heads: Çoklu dikkat (multi-head attention) sayısı.
        num_layers: Transformer encoder katman sayısı.
        dropout: Dropout oranı.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        embed = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embed = embed.transpose(0, 1)  # (seq_length, batch_size, embedding_dim)
        embed = self.positional_encoding(embed)
        transformer_output = self.transformer(embed)  # (seq_length, batch_size, embedding_dim)
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, seq_length, embedding_dim)
        output = self.fc(transformer_output)  # (batch_size, seq_length, vocab_size)
        return output

# GPT benzeri model - daha güçlü bir seçenek
class GPTModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # GPT-style decoder block (unidirectional/casual attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Create a square attention mask to ensure autoregressive property
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1000, 1000) * float('-inf'), diagonal=1)
        )
        
    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        seq_length = x.size(1)
        
        embed = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embed = embed.transpose(0, 1)  # (seq_length, batch_size, embedding_dim)
        embed = self.positional_encoding(embed)
        
        # Apply attention mask
        mask = self.mask[:seq_length, :seq_length]
        
        # Use zeros as memory since we're using decoder-only architecture
        memory = torch.zeros_like(embed)
        
        output = self.transformer(embed, memory, tgt_mask=mask)
        output = output.transpose(0, 1)
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return output

# RNN-GRU bazlı model
class GRUModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=64, hidden_dim=128, num_layers=1):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.gru(embed, hidden)
        output = self.fc(output)
        return output, hidden

# Model fabrikası - modelleri oluşturup döndüren fonksiyon
def create_model(model_type, model_params):
    """
    Belirtilen türde bir model oluşturur.
    
    Parametreler:
    model_type: Model türü ('lstm', 'transformer', 'gpt', 'gru')
    model_params: Model parametrelerini içeren sözlük
    """
    if model_type == 'lstm':
        return LSTMModel(
            vocab_size=model_params.get('vocab_size', 128),
            embedding_dim=model_params.get('embedding_dim', 64),
            hidden_dim=model_params.get('hidden_dim', 128),
            num_layers=model_params.get('num_layers', 1)
        )
    elif model_type == 'transformer':
        return TransformerModel(
            vocab_size=model_params.get('vocab_size', 128),
            embedding_dim=model_params.get('embedding_dim', 128),
            num_heads=model_params.get('num_heads', 4),
            num_layers=model_params.get('num_layers', 2),
            dropout=model_params.get('dropout', 0.1)
        )
    elif model_type == 'gpt':
        return GPTModel(
            vocab_size=model_params.get('vocab_size', 128),
            embedding_dim=model_params.get('embedding_dim', 128),
            num_heads=model_params.get('num_heads', 4),
            num_layers=model_params.get('num_layers', 3),
            dropout=model_params.get('dropout', 0.1)
        )
    elif model_type == 'gru':
        return GRUModel(
            vocab_size=model_params.get('vocab_size', 128),
            embedding_dim=model_params.get('embedding_dim', 64),
            hidden_dim=model_params.get('hidden_dim', 128),
            num_layers=model_params.get('num_layers', 1)
        )
    else:
        raise ValueError(f"Bilinmeyen model türü: {model_type}")

# Model kaydetme fonksiyonu
def save_model(model, model_type, model_params, epoch, optimizer, loss, save_path):
    """
    Modeli ve ilgili durumu kaydeder.
    """
    # models dizini yoksa oluştur
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Modelin durumunu, optimizer durumunu ve diğer meta verileri kaydet
    checkpoint = {
        'model_type': model_type,
        'model_params': model_params,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model başarıyla kaydedildi: {save_path}")
    return save_path

# Model yükleme fonksiyonu
def load_model(load_path, device='cpu'):
    """
    Kaydedilmiş bir modeli ve durumunu yükler.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Model meta verilerini al
    model_type = checkpoint['model_type']
    model_params = checkpoint['model_params']
    
    # Modeli oluştur
    model = create_model(model_type, model_params)
    
    # Model durumunu yükle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model başarıyla yüklendi: {load_path}")
    logger.info(f"Model türü: {model_type}, Parametreler: {model_params}")
    
    return model, checkpoint

# Eğitim döngüsü fonksiyonu
def train(model, dataloader, model_type, model_params, num_epochs=10, lr=0.001, device='cpu', save_dir='models', checkpoint_interval=1):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Model adı oluşturma (model türü ve parametrelerden)
    model_name = f"{model_type}"
    
    # Toplam iterasyon sayısı için progress bar
    total_batches = len(dataloader) * num_epochs
    with tqdm(total=total_batches, desc="Toplam İlerleme", unit="iter") as pbar_total:
        for epoch in range(num_epochs):
            total_loss = 0
            # Epoch bazlı progress bar
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="iter") as pbar_epoch:
                for batch_idx, (input_seq, target_seq) in enumerate(pbar_epoch):
                    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                    optimizer.zero_grad()
                    
                    # Model tipine göre ileri geçiş
                    if model_type in ['lstm', 'gru']:
                        output, _ = model(input_seq)
                    else:
                        output = model(input_seq)
                        
                    # Kayıp hesaplama
                    if model_type in ['lstm', 'gru', 'transformer']:
                        loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                    else:
                        loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Progress bar güncellemeleri
                    pbar_epoch.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': optimizer.param_groups[0]['lr']
                    })
                    pbar_total.update(1)
                    
            # Epoch sonu istatistikleri
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} tamamlandı ➔ Ortalama Kayıp: {avg_loss:.4f}")
            
            # Belirli aralıklarla modeli kaydet
            if (epoch + 1) % checkpoint_interval == 0:
                # Checkpoint dosya adı oluştur
                checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pt")
                save_model(model, model_type, model_params, epoch+1, optimizer, avg_loss, checkpoint_path)
                
    # Son modeli kaydet
    final_path = os.path.join(save_dir, f"{model_name}_final.pt")
    save_model(model, model_type, model_params, num_epochs, optimizer, avg_loss, final_path)
    
    return final_path

# Müzik üretim fonksiyonu: Başlangıç dizisi verilip belirli uzunlukta müzik oluşturulur.
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

# Config dosyasından ayarları yükleme fonksiyonu
def load_config(config_path):
    """
    Config dosyasından parametreleri yükler.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config dosyası bulunamadı: {config_path}")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config

# Enstrüman listesini alma
def get_instrument_list():
    """
    MIDI program numaralarına karşılık gelen enstrüman listesini döndürür.
    """
    instruments = {
        0: "Acoustic Grand Piano",
        1: "Bright Acoustic Piano",
        2: "Electric Grand Piano",
        4: "Electric Piano 1",
        5: "Electric Piano 2",
        6: "Harpsichord",
        7: "Clavinet",
        8: "Celesta",
        9: "Glockenspiel",
        13: "Marimba",
        14: "Xylophone",
        19: "Church Organ",
        20: "Reed Organ",
        24: "Acoustic Guitar (nylon)",
        25: "Acoustic Guitar (steel)",
        26: "Electric Guitar (jazz)",
        27: "Electric Guitar (clean)",
        32: "Acoustic Bass",
        33: "Electric Bass (finger)",
        40: "Violin",
        41: "Viola",
        42: "Cello",
        43: "Contrabass",
        46: "Harp",
        47: "Timpani",
        56: "Trumpet",
        57: "Trombone",
        58: "Tuba",
        60: "French Horn",
        68: "Oboe",
        69: "English Horn",
        70: "Bassoon",
        71: "Clarinet",
        72: "Piccolo",
        73: "Flute",
    }
    return instruments

# Config dosya örneği oluştur
def create_default_config(config_path="config.ini"):
    """
    Varsayılan config dosyası oluşturur.
    """
    config = configparser.ConfigParser()
    
    # Eğitim ayarları
    config['TRAINING'] = {
        'data_dir': './maestro-v3.0.0/2014',
        'model_type': 'transformer',  # lstm, transformer, gpt, gru
        'seq_length': '50',
        'batch_size': '64',
        'num_epochs': '10',
        'learning_rate': '0.0001',
        'save_dir': 'models',
        'checkpoint_interval': '2'
    }
    
    # Model parametreleri
    config['MODEL_PARAMS'] = {
        'vocab_size': '128',
        'embedding_dim': '128',
        'hidden_dim': '256',  # lstm, gru için
        'num_heads': '4',     # transformer, gpt için
        'num_layers': '2',
        'dropout': '0.1'      # transformer, gpt için
    }
    
    # Müzik üretim ayarları
    config['GENERATION'] = {
        'model_path': '',     # Boş bırakılırsa yeni model eğitilir
        'generation_length': '500',
        'temperature': '1.0',  # Yaratıcılık faktörü, >1 daha rastgele, <1 daha belirleyici
        'output_dir': 'generated'
    }
    
    # Müzik biçimlendirme
    config['MUSIC_FORMAT'] = {
        'bpm': '120',
        'note_duration': '0.3',
        'instrument_program': '0'  # 0 = Acoustic Grand Piano
    }
    
    # Config dosyasını kaydet
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    logger.info(f"Varsayılan config dosyası oluşturuldu: {config_path}")
    return config


def enhance_midi_quality(midi_path, output_path, bpm=120, instrument_program=0):
    """
    MIDI dosyasının kalitesini artırmak için daha gelişmiş parametreler ekler
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        logger.error(f"MIDI yüklenirken hata: {e}")
        return None

    # Yeni MIDI objesi oluştur
    enhanced_midi = pretty_midi.PrettyMIDI()
    new_instrument = pretty_midi.Instrument(program=instrument_program)

    # Nota parametrelerini iyileştirme
    for instr in midi.instruments:  # Dışarıdaki instrument'ı 'instr' olarak değiştirdik
        for note in instr.notes:
            # Rastgele velocity ekle (60-127 arası)
            velocity = random.randint(60, 127)
            
            # Nota süresine rastgele varyasyon ekle
            duration = note.end - note.start
            duration_variation = duration * random.uniform(0.9, 1.1)
            
            enhanced_note = pretty_midi.Note(
                velocity=velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.start + duration_variation
            )
            new_instrument.notes.append(enhanced_note)

    enhanced_midi.instruments.append(new_instrument)
    
    enhanced_midi.write(output_path)
    return output_path

def midi_to_wav(midi_path, wav_path, soundfont_path):
    """
    MIDI'yi WAV'a dönüştürür (FluidSynth gerektirir)
    """
    try:
        subprocess.run([
            'fluidsynth', '-ni', soundfont_path, 
            midi_path, '-F', wav_path, '-r', '44100', '-q'
        ], check=True, timeout=10)
        return wav_path
    except Exception as e:
        logger.error(f"WAV dönüşüm hatası: {e}")
        return None
    except FileNotFoundError:
        logger.error("FluidSynth kurulu değil! Lütfen kurulum yapın.")
        return None

def convert_to_mp3(wav_path, mp3_path, bitrate='192k'):
    """
    WAV dosyasını MP3'e dönüştürür
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        return mp3_path
    except Exception as e:
        logger.error(f"MP3 dönüşüm hatası: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Müzik Üretim Sistemi")
    parser.add_argument('--config', type=str, default='config.ini', help='Config dosyası yolu')
    parser.add_argument('--create_config', action='store_true', help='Varsayılan config dosyası oluştur')
    parser.add_argument('--list_instruments', action='store_true', help='Enstrüman listesini göster')
    parser.add_argument('--train', action='store_true', help='Yeni model eğit')
    parser.add_argument('--generate', action='store_true', help='Müzik üret')
    
    args = parser.parse_args()
    
    # Varsayılan config oluşturma
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Enstrüman listesini gösterme
    if args.list_instruments:
        instruments = get_instrument_list()
        print("\nKullanilabilir Enstrümanlar:")
        print("---------------------------")
        for program, name in instruments.items():
            print(f"{program}: {name}")
        return
    
    # Config dosyasını yükle
    if not os.path.exists(args.config):
        logger.warning(f"Config dosyası bulunamadı: {args.config}. Varsayılan config oluşturuluyor...")
        config = create_default_config(args.config)
    else:
        config = load_config(args.config)
    
    # Cihaz seçimi
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon için
    elif torch.cuda.is_available():
        device = 'cuda' # NVIDIA GPU için
    else:
        device = 'cpu'    
    logger.info(f"Kullanılan cihaz: {device}")
    
    # Eğitim parametreleri
    data_dir = config['TRAINING']['data_dir']
    model_type = config['TRAINING']['model_type']
    seq_length = int(config['TRAINING']['seq_length'])
    batch_size = int(config['TRAINING']['batch_size'])
    num_epochs = int(config['TRAINING']['num_epochs'])
    learning_rate = float(config['TRAINING']['learning_rate'])
    save_dir = config['TRAINING']['save_dir']
    checkpoint_interval = int(config['TRAINING']['checkpoint_interval'])
    
    # Model parametreleri
    model_params = {
        'vocab_size': int(config['MODEL_PARAMS']['vocab_size']),
        'embedding_dim': int(config['MODEL_PARAMS']['embedding_dim']),
        'hidden_dim': int(config['MODEL_PARAMS']['hidden_dim']),
        'num_heads': int(config['MODEL_PARAMS']['num_heads']),
        'num_layers': int(config['MODEL_PARAMS']['num_layers']),
        'dropout': float(config['MODEL_PARAMS']['dropout'])
    }
    
    # Üretim parametreleri
    model_path = config['GENERATION']['model_path']
    generation_length = int(config['GENERATION']['generation_length'])
    temperature = float(config['GENERATION']['temperature'])
    output_dir = config['GENERATION']['output_dir']
    
    # Müzik biçimlendirme
    bpm = int(config['MUSIC_FORMAT']['bpm'])
    soundfont_path = config["SOUND_SETTINGS"]["soundfont_path"]
    note_duration = float(config['MUSIC_FORMAT']['note_duration'])
    instrument_program = int(config['MUSIC_FORMAT']['instrument_program'])
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Eğitim modu
    if args.train or (not args.generate and not model_path):
        logger.info("Model eğitim modu başlatılıyor...")
        logger.info(f"Model türü: {model_type}, Veri dizini: {data_dir}")
        
        # Veri setini oluşturma ve DataLoader hazırlığı
        dataset = MusicDataset(data_dir, seq_length=seq_length)
        if len(dataset) == 0:
            logger.error("Veri setinde uygun MIDI dosyası bulunamadı!")
            return
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Modeli oluştur
        model = create_model(model_type, model_params)
        logger.info(f"Model oluşturuldu: {model_type}")
        
        # Modeli eğit
        final_model_path = train(
            model, dataloader, model_type, model_params, 
            num_epochs=num_epochs, 
            lr=learning_rate, 
            device=device,
            save_dir=save_dir,
            checkpoint_interval=checkpoint_interval
        )
        logger.info(f"Eğitim tamamlandı. Son model kaydedildi: {final_model_path}")

    # Üretim modu
    if args.generate:
        logger.info("Müzik üretim modu başlatılıyor...")
        
        # Model yolu belirtilmemişse son modeli kullan
        if not model_path:
            model_name = f"{model_type}"
            model_path = os.path.join(save_dir, f"{model_name}_final.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Kaydedilmiş model bulunamadı: {model_path}")

        # Modeli yükle
        model, checkpoint = load_model(model_path, device)
        model_type = checkpoint['model_type']
        
        start_seq = [60, 62, 64, 65, 67, 69, 71]
        logger.info(f"Başlangıç dizisi: {start_seq}")
        
        # Müzik üret
        generated_seq = generate_music(
            model,
            model_type,
            start_seq,
            length=generation_length,
            device=device,
            temperature=temperature
        )
        
        # MIDI'ye dönüştür ve kaydet
        output_path = os.path.join(output_dir, f"generated_{model_type}.mid")
        sequence_to_midi(
            generated_seq,
            file_path=output_path,
            bpm=bpm,
            note_duration=note_duration,
            instrument_program=instrument_program
        )
        logger.info(f"Müzik başarıyla oluşturuldu: {output_path}")

        enhanced_midi_path = os.path.join(output_dir, f"enhanced_{model_type}.mid")
        enhance_midi_quality(
            output_path,
            enhanced_midi_path,
            bpm=bpm,
            instrument_program=instrument_program
        )
        logger.info(f"Geliştirilmiş MIDI oluşturuldu: {enhanced_midi_path}")
        
        # WAV'a dönüştür
        wav_path = os.path.join(output_dir, f"output_{model_type}.wav")
        if not midi_to_wav(enhanced_midi_path, wav_path, soundfont_path):
            logger.info(f"Geliştirilmiş WAV oluşturulamaz: {wav_path}")
            return
        logger.info(f"Geliştirilmiş WAV oluşturuldu: {wav_path}")

        # MP3'e dönüştür
        mp3_path = os.path.join(output_dir, f"output_{model_type}.mp3")
        if convert_to_mp3(wav_path, mp3_path):
            logger.info(f"MP3 dosyası başarıyla kaydedildi: {mp3_path}")


if __name__ == "__main__":
    main()