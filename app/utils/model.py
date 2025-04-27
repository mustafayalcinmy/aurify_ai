

import logging
import os

import torch
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.gpt import GPTModel
from models.gru import GRUModel


logger = logging.getLogger("MusicGen")

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