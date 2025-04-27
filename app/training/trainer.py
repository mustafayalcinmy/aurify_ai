import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from utils.model import save_model
import logging


logger = logging.getLogger("MusicGen")
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
