# app/utils/model.py dosyasını güncelle

import logging
import os
import torch
import torch.nn as nn # weights_init için eklendi

# Modelleri import et
from app.models.lstm import LSTMModel
from app.models.transformer import TransformerModel
from app.models.gpt import GPTModel
from app.models.gru import GRUModel
from app.models.gan_generator import Generator
from app.models.gan_discriminator import Discriminator

logger = logging.getLogger("MusicGenModel") # Logger adı değişebilir

# --- Sıralı Model Fonksiyonları ---
def create_sequence_model(model_type, params):
    """
    Belirtilen türde ve parametrelerle Sıralı model oluşturur.
    Args:
        model_type (str): 'lstm', 'gru', 'transformer', 'gpt'
        params (dict): İlgili modelin constants dosyasından gelen parametre sözlüğü.
    """
    logger.info(f"Sıralı model oluşturuluyor: {model_type}")
    vocab_size = params.get('vocab_size') # Parametreleri al
    embedding_dim = params.get('embedding_dim')
    hidden_dim = params.get('hidden_dim')
    num_layers = params.get('num_layers')
    num_heads = params.get('num_heads')
    dropout = params.get('dropout')

    if model_type == 'lstm':
        if not all([vocab_size, embedding_dim, hidden_dim, num_layers]): raise ValueError("LSTM için eksik parametre!")
        return LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    elif model_type == 'gru':
        if not all([vocab_size, embedding_dim, hidden_dim, num_layers]): raise ValueError("GRU için eksik parametre!")
        return GRUModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    elif model_type == 'transformer':
        if not all([vocab_size, embedding_dim, num_heads, num_layers, dropout]): raise ValueError("Transformer için eksik parametre!")
        return TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, dropout)
    elif model_type == 'gpt':
         if not all([vocab_size, embedding_dim, num_heads, num_layers, dropout]): raise ValueError("GPT için eksik parametre!")
         return GPTModel(vocab_size, embedding_dim, num_heads, num_layers, dropout)
    else:
        raise ValueError(f"Bilinmeyen sıralı model türü: {model_type}")

def save_sequence_model(model, params, epoch, optimizer, loss, save_path):
    """Sıralı Modeli ve ilgili durumu kaydeder."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_params': params, # Sabitlerden gelen parametreleri kaydet
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Sıralı model başarıyla kaydedildi: {save_path}")
    return save_path

def load_sequence_model(load_path, device='cpu'):
    """
    Kaydedilmiş bir Sıralı modeli ve durumunu yükler.
    Modeli yeniden oluşturmak için checkpoint içindeki parametreleri kullanır.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Sıralı model dosyası bulunamadı: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    model_params = checkpoint['model_params'] # Kaydedilmiş parametreleri al

    # model_type parametresini bulmaya çalış
    # Eğer model_params içinde 'model_type' anahtarı yoksa, model tipini belirlemek zor olabilir.
    # Bu yüzden save_sequence_model içinde model_type'ı da kaydetmek iyi bir fikir olabilir.
    # Veya dosya adından çıkarmaya çalışılabilir (daha az güvenilir).
    # Şimdilik, parametrelerin hangi modele ait olduğunu bildiğimizi varsayalım veya model_params'a ekleyelim.
    model_type = model_params.get('model_type', None) # Örnek: save_sequence_model'da eklediğimizi varsayalım
    if not model_type:
        # Dosya adından çıkarmayı dene (basit bir varsayım)
        filename = os.path.basename(load_path)
        if 'lstm' in filename: model_type = 'lstm'
        elif 'gru' in filename: model_type = 'gru'
        elif 'transformer' in filename: model_type = 'transformer'
        elif 'gpt' in filename: model_type = 'gpt'
        else: raise ValueError(f"Model tipi checkpoint'te veya dosya adında bulunamadı: {load_path}")
        logger.warning(f"Model tipi checkpoint'te bulunamadı, dosya adından tahmin edildi: {model_type}")

    model = create_sequence_model(model_type, model_params) # Kaydedilmiş parametrelerle modeli oluştur
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logger.info(f"Sıralı model başarıyla yüklendi: {load_path}")
    logger.info(f"Model türü: {model_type}, Parametreler: {model_params}")

    return model, checkpoint


# --- GAN Model Fonksiyonları ---
def create_gan_models(generator_params, discriminator_params, pianoroll_params, gan_training_params):
    """
    GAN Generator ve Discriminator modellerini oluşturur.
    Args:
        generator_params (dict): GENERATOR_PARAMS sabitleri.
        discriminator_params (dict): DISCRIMINATOR_PARAMS sabitleri.
        pianoroll_params (dict): PIANOROLL_PARAMS sabitleri.
        gan_training_params (dict): GAN_TRAINING_PARAMS sabitleri.
    """
    latent_dim = gan_training_params['latent_dim']
    seq_length = pianoroll_params['seq_length']
    num_pitches = pianoroll_params['num_pitches']
    output_channels = generator_params['output_channels']
    input_channels = discriminator_params['input_channels']

    logger.info(f"GAN modelleri oluşturuluyor - LatentDim: {latent_dim}, SeqLen: {seq_length}, Pitches: {num_pitches}")

    netG = Generator(latent_dim=latent_dim, output_channels=output_channels, seq_length=seq_length, num_pitches=num_pitches)
    netD = Discriminator(input_channels=input_channels, seq_length=seq_length, num_pitches=num_pitches)

    # Ağırlık başlatma (isteğe bağlı)
    netG.apply(weights_init)
    netD.apply(weights_init)

    return netG, netD

def weights_init(m): # Yardımcı fonksiyon
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_gan_models(netG, netD, optimG, optimD, epoch, save_path, **kwargs):
    """GAN Modellerini ve optimizer durumlarını kaydeder."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Sabitleri kaydetmeye gerek yok, çünkü yüklerken constants'dan alacağız.
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizer_G_state_dict': optimG.state_dict(),
        'optimizer_D_state_dict': optimD.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)
    logger.info(f"GAN Checkpoint kaydedildi: {save_path}")

def load_gan_models(load_path, generator_params, discriminator_params, pianoroll_params, gan_training_params, device='cpu'):
    """
    Kaydedilmiş GAN modellerini (G ve D) yükler.
    Modelleri yeniden oluşturmak için constants dosyalarından gelen parametreleri kullanır.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"GAN model dosyası bulunamadı: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    # Sabitlerden alınan parametrelerle modelleri oluştur
    netG, netD = create_gan_models(generator_params, discriminator_params, pianoroll_params, gan_training_params)

    # Durumları yükle
    if 'generator_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['generator_state_dict'])
    else:
        logger.warning(f"Checkpoint dosyasında 'generator_state_dict' bulunamadı: {load_path}")

    if 'discriminator_state_dict' in checkpoint:
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
         logger.warning(f"Checkpoint dosyasında 'discriminator_state_dict' bulunamadı: {load_path}")

    netG.to(device)
    netD.to(device)

    logger.info(f"GAN Modelleri başarıyla yüklendi: {load_path}")

    # Optimizer durumları vb. için tüm checkpoint'i döndür
    return netG, netD, checkpoint