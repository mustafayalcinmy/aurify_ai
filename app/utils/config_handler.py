# app/utils/config_handler.py

import configparser
import os
import logging

logger = logging.getLogger("MusicGenConfig") # Logger adı değişebilir

def load_config(config_path):
    """Config dosyasını yükler."""
    if not os.path.exists(config_path):
        logger.error(f"Config dosyası bulunamadı: {config_path}")
        return None
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        logger.info(f"Config dosyası yüklendi: {config_path}")
        return config
    except configparser.Error as e:
        logger.error(f"Config dosyası okunurken hata: {e}")
        return None

def get_instrument_list():
    # ... (Bu fonksiyon aynı kalabilir) ...
    pass

def create_default_config(config_path="app/config.ini"): # Varsayılan yolu güncelledim
    """Varsayılan (sadeleştirilmiş) config dosyası oluşturur."""
    config = configparser.ConfigParser()

    config['EXPERIMENT'] = {
        'mode': 'sequence',
        'sequence_model_type': 'transformer'
    }

    config['PATHS'] = {
        'data_dir': './data/maestro-v3.0.0/2014', # Örnek yol
        'sequence_save_dir': 'models_sequence',
        'sequence_model_load_path': '',
        'gan_save_dir': 'models_gan',
        'gan_model_load_path': '',
        'generation_output_dir': 'generated',
        'visualization_output_dir': 'dataset_visualizations',
        'soundfont_path': '/path/to/your/soundfont.sf2' # Kullanıcı kendi yolunu girmeli
    }

    config['TRAINING'] = {
        'epochs': '50',
        'batch_size': '64',
        'checkpoint_interval': '5',
        'sequence_learning_rate': '0.0001'
    }

    config['GENERATION'] = {
        'sequence_generation_length': '500',
        'sequence_temperature': '1.0',
        # 'gan_num_samples': '5' # Opsiyonel override
    }

    config['MUSIC_FORMAT'] = {
        'bpm': '120',
        'instrument_program': '0',
        'sequence_note_duration': '0.3',
        # 'gan_velocity_threshold': '0.1' # Opsiyonel override
    }

    # [SOUND_SETTINGS] bölümü kaldırıldı, PATHS altına alındı. İsterseniz ayrı tutabilirsiniz.

    os.makedirs(os.path.dirname(config_path), exist_ok=True) # Klasör yoksa oluştur
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    logger.info(f"Varsayılan (sadeleştirilmiş) config dosyası oluşturuldu: {config_path}")
    return config