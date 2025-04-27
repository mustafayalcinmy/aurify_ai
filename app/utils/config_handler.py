# Config dosyasından ayarları yükleme fonksiyonu
import configparser
import os
import logging

logger = logging.getLogger("MusicGen")


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
    
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    logger.info(f"Varsayılan config dosyası oluşturuldu: {config_path}")
    return config