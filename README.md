```markdown
# AURIFY

Bu proje, LSTM, Transformer, GPT ve GRU gibi derin öğrenme modellerini kullanarak otomatik müzik üretimi sağlayan bir sistemdir. MIDI dosyaları üretir ve bunları WAV/MP3 formatlarına dönüştürür.

## Özellikler

- 🎹 Çoklu model desteği (LSTM, Transformer, GPT, GRU)
- 🎼 MIDI'den diziye ve diziden MIDI'ye dönüşüm
- 🎧 Üretilen MIDI'leri WAV ve MP3'e dönüştürme
- ⚙️ Yapılandırılabilir parametreler (config.ini)
- 🎚️ Sıcaklık kontrollü üretim
- 🎻 Çoklu enstrüman desteği

## Kurulum

1. Gereksinimler:
```bash
pip install -r requirements.txt
```

2. FluidSynth kurulumu (Ses dönüşümleri için):
```bash
# macOS
brew install fluidsynth

# Linux
sudo apt-get install fluidsynth
```

3. SoundFont indirin (ör. `GeneralUser.sf2`) ve `config.ini`'de yolunu belirtin

## Kullanım

### Temel Komutlar

- Varsayılan config dosyası oluştur:
```bash
python main.py --create_config
```

- Model eğit:
```bash
python main.py --train
```

- Müzik üret:
```bash
python main.py --generate
```

- Enstrüman listesini göster:
```bash
python main.py --list_instruments
```

### Yapılandırma

`config.ini` dosyasından ayarları özelleştirebilirsiniz:

```ini
[TRAINING]
data_dir = ./data
model_type = transformer  # lstm, transformer, gpt, gru
num_epochs = 20

[MODEL_PARAMS]
hidden_dim = 256
num_heads = 8

[GENERATION]
generation_length = 500
temperature = 0.9  # 0.1-2.0 arası

[MUSIC_FORMAT]
instrument_program = 0  # 0=Piano, 40=Violin, vb.
```

## Model Eğitimi

1. Veri kümesini `data_dir` dizinine yerleştirin
2. Config dosyasında model parametrelerini ayarlayın
3. Eğitimi başlatın:
```bash
python main.py --train
```

Eğitilen modeller `models/` dizinine kaydedilir.

## Müzik Üretimi

1. Eğitilmiş modeli seçin (`config.ini`'de `model_path`)
2. Üretim parametrelerini ayarlayın
3. Müzik oluşturun:
```bash
python main.py --generate
```

Çıktılar `generated/` dizininde:
- MIDI (.mid)
- İşlenmiş WAV/MP3
- Geliştirilmiş MIDI
