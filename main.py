import configparser
from datetime import datetime
import glob
import os
import time
import torch
from torch.utils.data import DataLoader
import logging
import argparse

from tqdm import tqdm
from app.utils.config_handler import load_config, create_default_config
from app.training.trainer import train
from app.utils.audio_converter import sequence_to_midi, midi_to_wav, convert_to_mp3, enhance_midi_quality, visualize_midi_piano_roll
from app.data.dataset import MusicDataset
from app.utils.config_handler import get_instrument_list

# Config Yöneticisi
from app.utils.config_handler import load_config, create_default_config, get_instrument_list

# Model Yöneticileri (Sıralı ve GAN)
from app.utils.model import (
    create_sequence_model, load_sequence_model, save_sequence_model,
    create_gan_models, load_gan_models, save_gan_models
)

# Üretim Fonksiyonları (Sıralı ve GAN)
from app.generation.generator import generate_music, generate_music_gan

# Eğitim Fonksiyonları (Sıralı ve GAN)
from app.training.trainer import train as train_sequence
try:
    from app.training.gan_trainer import train_gan
except ImportError:
    train_gan = None

# Ses/MIDI Dönüştürücüler
from app.utils.audio_converter import (
    sequence_to_midi, midi_to_wav, convert_to_mp3, enhance_midi_quality,
    visualize_midi_piano_roll, pianoroll_tensor_to_midi
)

# Veri Setleri (Sıralı ve GAN)
from app.data.dataset import MusicDataset, PianoRollDataset

# Model Sabitleri
from app.models.constants_sequence import SEQUENCE_MODEL_PARAMS
from app.models.constants_gan import (
    GAN_TRAINING_PARAMS, PIANOROLL_PARAMS, GENERATOR_PARAMS,
    DISCRIMINATOR_PARAMS, GAN_GENERATION_DEFAULTS
)

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MusicGen")


def visualize_entire_dataset(config):
    """
    Config dosyasında belirtilen data_dir içindeki tüm MIDI dosyalarını görselleştirir.
    """
    data_dir = config.get('PATHS', 'data_dir', fallback=None)
    vis_output_dir = config.get('PATHS', 'visualization_output_dir', fallback='dataset_visualizations')
   
    if not data_dir or not os.path.isdir(data_dir):
        logger.error(f"Geçerli bir veri seti dizini bulunamadı veya belirtilmedi: {data_dir}")
        return

    os.makedirs(vis_output_dir, exist_ok=True)
    logger.info(f"Veri seti görselleştirmeleri '{vis_output_dir}' klasörüne kaydedilecek.")

    # .mid ve .midi dosyalarını bul (alt klasörler dahil)
    midi_files = glob.glob(os.path.join(data_dir, '**', '*.mid'), recursive=True)
    midi_files.extend(glob.glob(os.path.join(data_dir, '**', '*.midi'), recursive=True))

    if not midi_files:
        logger.warning(f"'{data_dir}' içinde görselleştirilecek MIDI dosyası bulunamadı.")
        return

    logger.info(f"{len(midi_files)} adet MIDI dosyası görselleştirilecek...")

    processed_count = 0
    error_count = 0
    # tqdm ile ilerleme çubuğu
    for midi_file in tqdm(midi_files, desc="MIDI Görselleştiriliyor"):
        try:
            # Çıktı PNG dosyasının adını oluştur (orijinal dosya adını koruyarak)
            relative_path = os.path.relpath(midi_file, data_dir)
            base_name = os.path.splitext(relative_path)[0]
            # Klasör yapısını korumak için alt dizinleri oluştur
            output_png_dir = os.path.join(vis_output_dir, os.path.dirname(relative_path))
            os.makedirs(output_png_dir, exist_ok=True)
            output_png_path = os.path.join(vis_output_dir, base_name + ".png")

            # Her MIDI dosyasının ilk enstrümanını görselleştir (isteğe bağlı olarak değiştirilebilir)
            success = visualize_midi_piano_roll(midi_file, output_png_path=output_png_path, instrument_index=0)
            if success:
                processed_count += 1
            else:
                 error_count += 1
        except Exception as e:
            logger.error(f"'{midi_file}' görselleştirilirken beklenmedik hata: {e}", exc_info=False) # Detaylı log için True yap
            error_count += 1

    logger.info(f"Veri seti görselleştirme tamamlandı. Başarılı: {processed_count}, Hatalı: {error_count}")



def main():
    parser = argparse.ArgumentParser(description="Müzik Üretim Sistemi")
    parser.add_argument('--config', type=str, default='app/config.ini', help='Config dosyası yolu')
    parser.add_argument('--create_config', action='store_true', help='Varsayılan config dosyası oluştur')
    parser.add_argument('--list_instruments', action='store_true', help='Enstrüman listesini göster')
    parser.add_argument('--train', action='store_true', help='Configde belirtilen moda göre model eğit')
    parser.add_argument('--generate', action='store_true', help='Configde belirtilen moda göre müzik üret')
    parser.add_argument('--visualize_dataset', action='store_true', help='Tüm veri setini piyano rulosu olarak görselleştir')

    args = parser.parse_args()

    # --- Config İşlemleri ---
    if args.create_config: create_default_config(args.config); return
    if args.list_instruments: get_instrument_list(); print("\n(Yukarıdaki listeden MUSIC_FORMAT -> instrument_program seçin)"); return # print eklendi

    config = load_config(args.config)
    if config is None: return

    # --- Mod ve Temel Ayarları Oku ---
    try:
        run_mode = config.get('EXPERIMENT', 'mode', fallback='sequence').lower()
        sequence_model_type = config.get('EXPERIMENT', 'sequence_model_type', fallback='transformer').lower()

        # Path'leri al
        data_dir = config.get('PATHS', 'data_dir')
        generation_output_dir = config.get('PATHS', 'generation_output_dir')
        soundfont_path = config.get('PATHS', 'soundfont_path')
        models_save_dir = config.get('PATHS', 'models_save_dir')
        model_load_path = config.get('PATHS', 'model_load_path', fallback=None)


        # Genel Eğitim Ayarları
        epochs = config.getint('TRAINING', 'epochs', fallback=50)
        batch_size = config.getint('TRAINING', 'batch_size', fallback=64)
        checkpoint_interval = config.getint('TRAINING', 'checkpoint_interval', fallback=10)

        # MIDI Format Ayarları
        bpm = config.getint('MUSIC_FORMAT', 'bpm', fallback=120)
        instrument_program = config.getint('MUSIC_FORMAT', 'instrument_program', fallback=0)

    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logger.error(f"Config dosyası okunurken hata: {e}. Lütfen '{args.config}' dosyasını kontrol edin.")
        return

    logger.info(f"Çalışma Modu: {run_mode.upper()}")
    if run_mode == 'sequence':
        logger.info(f"Sıralı Model Tipi: {sequence_model_type.upper()}")

    # --- Veri Seti Görselleştirme ---
    if args.visualize_dataset:
        logger.info("Veri seti görselleştirme modu başlatılıyor...")
        visualize_entire_dataset(config) # Config objesini direkt gönder
        return

    # --- Cihaz Seçimi ---
    if torch.backends.mps.is_available(): device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    logger.info(f"Kullanılan cihaz: {device}")

    # Çıktı dizinlerini oluştur
    os.makedirs(generation_output_dir, exist_ok=True)
    os.makedirs(models_save_dir, exist_ok=True)

    # --- EĞİTİM ---
    if args.train:
        logger.info(f"{run_mode.upper()} Modeli Eğitim Modu Başlatılıyor...")
        logger.info(f"Veri dizini: {data_dir}")

        if run_mode == 'sequence':
            # 1. Veri Seti
            seq_len_const = SEQUENCE_MODEL_PARAMS[sequence_model_type].get('seq_length', 50) # Sabitlerden veya default al
            dataset = MusicDataset(data_dir, seq_length=seq_len_const)
            if len(dataset) == 0: logger.error("Uygun sequence verisi bulunamadı!"); return
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # 2. Model
            model_params = SEQUENCE_MODEL_PARAMS.get(sequence_model_type)
            if not model_params: logger.error(f"Geçersiz sıralı model tipi: {sequence_model_type}"); return
            model = create_sequence_model(sequence_model_type, model_params)
            # 3. Eğitim
            lr = config.getfloat('TRAINING', 'sequence_learning_rate', fallback=0.0001)
            final_model_path = train_sequence(
                model, dataloader, sequence_model_type, model_params, # model_params'ı trainer'a veriyoruz
                num_epochs=epochs, lr=lr, device=device,
                save_dir=models_save_dir, checkpoint_interval=checkpoint_interval
            )
            logger.info(f"Sıralı model eğitimi tamamlandı. Son model: {final_model_path}")

        elif run_mode == 'gan':
            if train_gan is None: logger.error("GAN Trainer bulunamadı!"); return
            # 1. Veri Seti
            dataset = PianoRollDataset(
                data_dir,
                seq_length=PIANOROLL_PARAMS['seq_length'],
                fs=PIANOROLL_PARAMS['fs'],
                pitch_range=(PIANOROLL_PARAMS['pitch_min'], PIANOROLL_PARAMS['pitch_max']),
                include_velocity=PIANOROLL_PARAMS['include_velocity'],
                step_size=PIANOROLL_PARAMS['step_size']
            )
            if len(dataset) == 0: logger.error("Uygun pianoroll verisi bulunamadı!"); return
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            # 2. Modeller
            netG, netD = create_gan_models(GENERATOR_PARAMS, DISCRIMINATOR_PARAMS, PIANOROLL_PARAMS, GAN_TRAINING_PARAMS)
            # 3. Eğitim
            final_model_path, G_losses, D_losses = train_gan(
                generator=netG, discriminator=netD, dataloader=dataloader,
                num_epochs=epochs,
                lr_g=GAN_TRAINING_PARAMS['lr_g'], lr_d=GAN_TRAINING_PARAMS['lr_d'],
                beta1=GAN_TRAINING_PARAMS['beta1'], latent_dim=GAN_TRAINING_PARAMS['latent_dim'],
                device=device, save_dir=models_save_dir, checkpoint_interval=checkpoint_interval
            )
            logger.info(f"GAN eğitimi tamamlandı. Son model bilgisi: {final_model_path}")

    # --- ÜRETİM ---
    if args.generate:
        logger.info(f"{run_mode.upper()} Modeli Üretim Modu Başlatılıyor...")
        all_generated_midi_paths = [] # Üretilen tüm MIDI yollarını tutacak liste

        if run_mode == 'sequence':
            # 1. Model Yükle
            load_path = sequence_model_load_path
            if not load_path: # Yol boşsa son eğitileni bul
                # Model tipini config'den veya sabitlerden al
                # seq_model_type_for_load = config.get('EXPERIMENT', 'sequence_model_type', fallback='transformer').lower()
                # load_path = os.path.join(models_save_dir, f"{seq_model_type_for_load}_final.pt") # Dosya adını tahmin et
                # Daha güvenli yol: En son değiştirilen .pt dosyasını bulmak
                list_of_files = glob.glob(os.path.join(models_save_dir, '*.pt'))
                if not list_of_files:
                    logger.error(f"Sequence model yükleme yolu belirtilmemiş ve '{models_save_dir}' içinde model bulunamadı.")
                    return
                load_path = max(list_of_files, key=os.path.getctime)
                logger.info(f"Sequence model yükleme yolu belirtilmedi, en son model kullanılıyor: {load_path}")

            try:
                model, checkpoint = load_sequence_model(load_path, device)
                loaded_model_params = checkpoint['model_params']
                # sequence_model_type'ı checkpoint'ten almak daha güvenli olabilir
                sequence_model_type = loaded_model_params.get('model_type', sequence_model_type) # Update type if found
                logger.info(f"Checkpoint'ten yüklenen model tipi: {sequence_model_type.upper()}")

            except FileNotFoundError: logger.error(f"Sıralı model bulunamadı: {load_path}"); return
            except Exception as e: logger.error(f"Sıralı model yüklenirken hata: {e}"); return

            # 2. Üret
            start_seq = [60, 62, 64, 65, 67] # Örnek başlangıç (config'den alınabilir)
            gen_len = config.getint('GENERATION', 'sequence_generation_length', fallback=500)
            temp = config.getfloat('GENERATION', 'sequence_temperature', fallback=1.0)
            note_dur = config.getfloat('MUSIC_FORMAT', 'sequence_note_duration', fallback=0.3)

            generated_seq = generate_music(
                model, sequence_model_type, start_seq,
                length=gen_len, device=device, temperature=temp
            )
            # 3. MIDI Kaydet
            ts = int(time.time())
            output_midi_path = os.path.join(generation_output_dir, f"seq_{sequence_model_type}_{ts}.mid")
            sequence_to_midi(
                generated_seq, output_midi_path,
                bpm=bpm, note_duration=note_dur, instrument_program=instrument_program
            )
            logger.info(f"Sıralı model ile MIDI oluşturuldu: {output_midi_path}")
            all_generated_midi_paths.append(output_midi_path) # Listeye ekle

        elif run_mode == 'gan':
            # 1. Model Yükle (Sadece Generator)
            load_path = model_load_path
            # Önceki adımdaki NameError düzeltmesi için 'time' import edildiğini varsayıyoruz.
            # Eğer 'time' import edilmediyse, dosyanın başına 'import time' eklenmeli.

            # Model yükleme uyarısını düzeltmek için checkpoint dosyasını kullanmayı dene
            if not load_path:
                 # Son checkpoint'i bulmayı dene
                list_of_files = glob.glob(os.path.join(models_save_dir, 'gan_checkpoint_epoch_*.pt'))
                if not list_of_files:
                    logger.error(f"GAN model yükleme yolu belirtilmemiş ve '{models_save_dir}' içinde checkpoint bulunamadı.")
                    return
                load_path = max(list_of_files, key=os.path.getctime)
                logger.warning(f"GAN model yükleme yolu belirtilmedi, en son checkpoint kullanılıyor: {load_path}")
            # Eğer hala final.pt kullanmak istiyorsan ve uyarıları görmezden gelmek istersen:
            # elif not os.path.exists(load_path):
            #     logger.error(f"Belirtilen GAN modeli yolu bulunamadı: {load_path}")
            #     return

            try:
                 # Modeli yüklemek için sabitlere ihtiyacımız var
                 # `load_gan_models` fonksiyonu checkpoint yapısını bekler.
                 # Eğer `generator_final.pt` sadece state_dict içeriyorsa, bu hata verir.
                 # Yukarıdaki gibi son checkpoint'i yüklemek daha güvenlidir.
                netG, _, checkpoint_data = load_gan_models(
                    load_path, GENERATOR_PARAMS, DISCRIMINATOR_PARAMS,
                    PIANOROLL_PARAMS, GAN_TRAINING_PARAMS, device=device
                )
                logger.info(f"GAN modeli yüklendi (Epoch {checkpoint_data.get('epoch', 'Bilinmiyor')}).")

            except FileNotFoundError: logger.error(f"GAN modeli/checkpoint bulunamadı: {load_path}"); return
            # except KeyError as e: logger.error(f"GAN modeli yüklenirken eksik anahtar: {e}. Yüklenen dosya ({load_path}) beklenen checkpoint formatında olmayabilir."); return
            except Exception as e: logger.error(f"GAN modeli yüklenirken hata: {e}"); return

            # 2. Üret
            num_samples = config.getint('GENERATION', 'gan_num_samples', fallback=GAN_GENERATION_DEFAULTS['num_samples'])
            vel_threshold = config.getfloat('MUSIC_FORMAT', 'gan_velocity_threshold', fallback=GAN_GENERATION_DEFAULTS['velocity_threshold'])

            generated_midi_paths_gan = generate_music_gan(
                generator=netG,
                latent_dim=GAN_TRAINING_PARAMS['latent_dim'],
                num_samples=num_samples,
                device=device,
                output_dir=generation_output_dir,
                # pianoroll_tensor_to_midi için parametreler:
                fs=PIANOROLL_PARAMS['fs'],
                pitch_range=(PIANOROLL_PARAMS['pitch_min'], PIANOROLL_PARAMS['pitch_max']),
                velocity_threshold=vel_threshold,
                instrument_program=instrument_program,
                bpm=bpm
            )
            # 3. Kaydet (generate_music_gan içinde yapılıyor)
            if generated_midi_paths_gan:
                logger.info(f"GAN ile {len(generated_midi_paths_gan)} adet müzik oluşturuldu.")
                all_generated_midi_paths.extend(generated_midi_paths_gan) # Listeye ekle
            else:
                logger.warning("GAN ile müzik üretilemedi.")

        # --- Üretilen Tüm MIDI Dosyalarını WAV/MP3'e Dönüştür ---
        if not all_generated_midi_paths:
            logger.warning("MP3'e dönüştürülecek MIDI dosyası üretilmedi.")
            # return yerine geç, çünkü sadece generate çalıştırılmış olabilir
            # return

        elif not soundfont_path or not os.path.exists(soundfont_path):
            logger.warning(f"Soundfont dosyası bulunamadı veya geçerli değil: {soundfont_path}. WAV/MP3 dönüşümü atlanıyor.")
        else:
            logger.info("Üretilen MIDI dosyaları MP3 formatına dönüştürülüyor...")
            converted_count = 0
            for midi_path in all_generated_midi_paths:
                try:
                    # --- YENİ İSİMLENDİRME MANTIĞI ---
                    base_name = os.path.splitext(os.path.basename(midi_path))[0]
                    # Enhance (Opsiyonel)
                    # Not: Enhance edilmiş MIDI'yi ayrı kaydetmek yerine doğrudan WAV'a geçebiliriz
                    # veya aynı isimle kaydedip üzerine yazabiliriz. Şimdilik ayrı isimle devam edelim.
                    enhanced_midi_path = os.path.join(generation_output_dir, f"{base_name}_enhanced.mid")
                    if not enhance_midi_quality(midi_path, enhanced_midi_path, bpm=bpm, instrument_program=instrument_program):
                        logger.warning(f"MIDI kalitesi artırılamadı: {midi_path}, orijinal kullanılacak.")
                        enhanced_midi_path = midi_path # Orijinali kullan

                    # WAV'a dönüştür
                    wav_path = os.path.join(generation_output_dir, f"{base_name}.wav")
                    if midi_to_wav(enhanced_midi_path, wav_path, soundfont_path):
                        logger.info(f"WAV oluşturuldu: {wav_path}")
                        # MP3'e dönüştür
                        mp3_path = os.path.join(generation_output_dir, f"{base_name}.mp3")
                        if convert_to_mp3(wav_path, mp3_path):
                            logger.info(f"MP3 oluşturuldu: {mp3_path}")
                            converted_count += 1
                        else:
                            logger.error(f"MP3 dönüşümü başarısız: {wav_path}")
                        # İsteğe bağlı: WAV dosyasını sil
                        try:
                            os.remove(wav_path)
                            logger.debug(f"Geçici WAV dosyası silindi: {wav_path}")
                        except OSError as e:
                            logger.warning(f"WAV dosyası silinemedi: {e}")
                    else:
                        logger.error(f"WAV dönüşümü başarısız: {enhanced_midi_path}")

                    # İsteğe bağlı: Enhanced MIDI dosyasını sil (eğer orijinalden farklıysa)
                    if enhanced_midi_path != midi_path:
                         try:
                             os.remove(enhanced_midi_path)
                             logger.debug(f"Geçici Enhanced MIDI dosyası silindi: {enhanced_midi_path}")
                         except OSError as e:
                             logger.warning(f"Enhanced MIDI silinemedi: {e}")

                except Exception as e:
                    logger.error(f"'{midi_path}' dosyası işlenirken hata oluştu: {e}", exc_info=True) # Hata ayıklama için True

            logger.info(f"Toplam {converted_count} adet MIDI dosyası MP3'e dönüştürüldü.")

        # İsteğe bağlı: Üretilen MIDI'leri WAV/MP3'e çevir
        # ... (enhance_midi_quality, midi_to_wav, convert_to_mp3 çağrıları eklenebilir) ...


        # # WAV'a dönüştür
        # wav_path = os.path.join(output_dir, f"output_{model_type}.wav")
        # if not midi_to_wav(enhanced_midi_path, wav_path, soundfont_path):
        #     logger.info(f"Geliştirilmiş WAV oluşturulamaz: {wav_path}")
        #     return
        # logger.info(f"Geliştirilmiş WAV oluşturuldu: {wav_path}")

        # # MP3'e dönüştür
        # mp3_path = os.path.join(output_dir, f"output_{model_type}.mp3")
        # if convert_to_mp3(wav_path, mp3_path):
        #     logger.info(f"MP3 dosyası başarıyla kaydedildi: {mp3_path}")


if __name__ == "__main__":
    main()