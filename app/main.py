import os
import torch
from torch.utils.data import DataLoader
import logging
import argparse
from app.utils.config_handler import load_config, create_default_config
from utils.model import create_model, load_model
from generation.generator import generate_music
from training.trainer import train
from utils.audio_converter import sequence_to_midi, midi_to_wav, convert_to_mp3, enhance_midi_quality
from data.dataset import MusicDataset
from app.utils.config_handler import get_instrument_list

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MusicGen")


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