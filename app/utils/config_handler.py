import configparser
import os
import logging
import pretty_midi

logger = logging.getLogger("ConfigHandler")

def load_config(config_path='app/config.ini'):
    """
    Belirtilen yoldan config dosyasını okur.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config dosyası bulunamadı: {config_path}")
        logger.info(f"Varsayılan bir config dosyası oluşturmak için --create_config argümanını kullanın.")
        return None

    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        logger.info(f"Config dosyası başarıyla yüklendi: {config_path}")
        return config
    except configparser.Error as e:
        logger.error(f"Config dosyası okunurken hata oluştu: {e}")
        return None


def create_default_config(config_path='app/config.ini'):
    """
    Varsayılan ayarlarla bir config dosyası oluşturur veya üzerine yazar.
    """
    config = configparser.ConfigParser()

    # --- Deney Ayarları ---
    config['EXPERIMENT'] = {
        '# Çalışma modu: sequence (LSTM/GRU/Transformer/GPT) veya gan': '',
        'mode': 'sequence',
        '# Eğer mode = sequence ise kullanılacak model tipi: lstm, gru, transformer, gpt': '',
        'sequence_model_type': 'transformer'
    }

    # --- Dosya Yolları ---
    config['PATHS'] = {
        '# MIDI dosyalarının bulunduğu ana klasör': '',
        'data_dir': './datasets/maestro-v3.0.0',
        '# Modellerin kaydedileceği ve yükleneceği ana klasör': '',
        'models_save_dir': 'models',
        '# Üretim modunda yüklenecek spesifik model dosyasının yolu (isteğe bağlı, boş bırakılırsa en son kaydedilen model kullanılır)': '',
        'model_load_path': '', # Boş bırakılabilir, main.py en son modeli bulur
        '# Üretilen MIDI/MP3 dosyalarının kaydedileceği klasör': '',
        'generation_output_dir': 'generated',
        '# MIDI dosyalarını WAV/MP3\'e çevirmek için kullanılacak SoundFont dosyasının yolu': '',
        'soundfont_path': '/path/to/your/soundfont.sf2', # Kullanıcı kendi yolunu girmeli
        '# (Opsiyonel) Veri seti görselleştirmelerinin kaydedileceği klasör': '',
        '# visualization_output_dir': 'dataset_visualizations' # Opsiyonel olduğu için yorum satırı
    }

    # --- Eğitim Ayarları ---
    config['TRAINING'] = {
        '# Toplam eğitim epoch sayısı': '',
        'epochs': 50,
        '# Eğitim sırasında kullanılacak batch boyutu': '',
        'batch_size': 64,
        '# Kaç epoch\'ta bir model checkpoint\'inin kaydedileceği': '',
        'checkpoint_interval': 10,
        '# Sıralı modeller (sequence) için öğrenme oranı': '',
        'sequence_learning_rate': 0.0001,
        '# GAN modelleri için öğrenme oranları (gan_trainer.py içinde kullanılır)': '',
        '# gan_learning_rate_g': 0.0002, # Sabitlerden geliyor, config'e eklemeye gerek yok
        '# gan_learning_rate_d': 0.0002  # Sabitlerden geliyor, config'e eklemeye gerek yok
    }

    # --- Üretim Ayarları ---
    config['GENERATION'] = {
        '# Sıralı modeller için üretilecek nota dizisi uzunluğu': '',
        'sequence_generation_length': 500,
        '# Sıralı modeller için üretimdeki rastgelelik seviyesi (düşük = daha deterministik)': '',
        'sequence_temperature': 1.0,
        '# GAN modelleri için kaç adet örnek üretileceği': '',
        '# gan_num_samples': 5 # Opsiyonel, main.py'de varsayılanı var
    }

    # --- Müzik Formatı Ayarları ---
    config['MUSIC_FORMAT'] = {
        '# MIDI dosyaları için tempo (BPM - Beats Per Minute)': '',
        'bpm': 120,
        '# MIDI dosyalarında kullanılacak enstrüman program numarası (0-127, GM standardı)': '',
        '# (Enstrüman listesi için --list_instruments kullanın)': '',
        'instrument_program': 0, # 0: Acoustic Grand Piano
        '# Sıralı modellerin ürettiği notaların saniye cinsinden varsayılan süresi': '',
        'sequence_note_duration': 0.3,
        '# GAN modelleri için piyano rulosundan MIDIye çevirirken velocity eşik değeri': '',
        '# gan_velocity_threshold': 0.1 # Opsiyonel, main.py'de varsayılanı var
    }

    try:
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        logger.info(f"Varsayılan config dosyası başarıyla oluşturuldu: {config_path}")
    except IOError as e:
        logger.error(f"Config dosyası yazılamadı: {e}")


def get_instrument_list():
    """
    pretty_midi kütüphanesindeki standart General MIDI enstrüman listesini gösterir.
    """
    print("General MIDI Enstrüman Listesi (Program Numarası: İsim):")
    print("-" * 50)
    for i in range(128):
        instrument_name = pretty_midi.program_to_instrument_name(i)
        print(f"{i}: {instrument_name}")
    print("-" * 50)


# Bu dosya doğrudan çalıştırıldığında varsayılan config oluşturmayı veya
# enstrüman listesini göstermeyi sağlayabilir (opsiyonel)
if __name__ == "__main__":
    # Örnek kullanım:
    # python app/utils/config_handler.py --create
    # python app/utils/config_handler.py --list
    import argparse
    parser = argparse.ArgumentParser(description="Config Handler Yardımcı Script")
    parser.add_argument('--create', action='store_true', help='Varsayılan config.ini dosyasını oluşturur.')
    parser.add_argument('--list', action='store_true', help='General MIDI enstrüman listesini gösterir.')
    parser.add_argument('--path', type=str, default='app/config.ini', help='İşlem yapılacak config dosyasının yolu.')

    args = parser.parse_args()

    if args.create:
        create_default_config(args.path)
    elif args.list:
        get_instrument_list()
    else:
        print("Lütfen bir işlem belirtin: --create veya --list")