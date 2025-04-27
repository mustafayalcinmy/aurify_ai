# app/training/gan_trainer.py (Yeni Dosya)

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import logging
import time

# Model kaydetme fonksiyonunu utils'den import et
from app.utils.model import save_gan_models

# Logger'ı ayarla (main.py'deki ile aynı ismi kullanabilir veya farklı olabilir)
logger = logging.getLogger("MusicGenGAN")

def train_gan(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    lr_g,
    lr_d,
    beta1,
    latent_dim,
    device,
    save_dir,
    checkpoint_interval,
    ):
    """
    GAN (Generator ve Discriminator) modelini eğitir.

    Args:
        generator (nn.Module): Generator modeli.
        discriminator (nn.Module): Discriminator modeli.
        dataloader (DataLoader): Eğitim verisini sağlayan DataLoader.
        num_epochs (int): Toplam epoch sayısı.
        lr_g (float): Generator optimizer öğrenme oranı.
        lr_d (float): Discriminator optimizer öğrenme oranı.
        beta1 (float): Adam optimizer için beta1 parametresi.
        latent_dim (int): Gürültü vektörü boyutu.
        device (str): Eğitim için kullanılacak cihaz ('cpu', 'cuda', 'mps').
        save_dir (str): Modellerin ve checkpoint'lerin kaydedileceği klasör.
        checkpoint_interval (int): Kaç epoch'ta bir checkpoint kaydedileceği.
    """

    # Modelleri cihaza taşı
    generator.to(device)
    discriminator.to(device)

    # Kayıp Fonksiyonu (Genellikle Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Optimizatörler (DCGAN makalesindeki gibi Adam önerilir)
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))

    # Gerçek ve Sahte etiketler (Discriminator eğitimi için)
    real_label = 1.
    fake_label = 0.

    # İlerleme ve kayıpları takip etmek için listeler
    G_losses = []
    D_losses = []
    iters = 0

    logger.info("GAN Eğitimi Başlatılıyor...")
    logger.info(f"Epochs: {num_epochs}, Batch Size: {dataloader.batch_size}, Latent Dim: {latent_dim}, Device: {device}")
    logger.info(f"LR_G: {lr_g}, LR_D: {lr_d}, Beta1: {beta1}")
    logger.info(f"Save Directory: {save_dir}, Checkpoint Interval: {checkpoint_interval}")

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_D_loss_total = 0.0
        epoch_G_loss_total = 0.0
        discriminator.train() # Modelleri train moduna al
        generator.train()

        # DataLoader üzerinden iterasyon (tqdm ile ilerleme çubuğu)
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar_epoch:
            for i, real_data in enumerate(pbar_epoch):

                # Veriyi cihaza taşı (real_data'nın doğru formatta geldiğini varsayıyoruz - örn: (batch, 1, seq, pitch))
                real_data = real_data.to(device)
                b_size = real_data.size(0) # Mevcut batch boyutunu al

                # ---------------------
                #  Discriminator Eğitimi
                # ---------------------
                discriminator.zero_grad()

                # 1. Gerçek Veri ile: Discriminator'ın gerçek veriyi tanımasını sağla
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output_real = discriminator(real_data).view(-1) # Çıktıyı (batch_size,) yap
                errD_real = criterion(output_real, label)
                if torch.isnan(errD_real): logger.warning(f"NaN detected in errD_real at epoch {epoch+1}, iter {i}"); continue
                errD_real.backward()
                D_x = output_real.mean().item() # Gerçek verinin ortalama D skoru

                # 2. Sahte Veri ile: Discriminator'ın sahte veriyi tanımasını sağla
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device) # Gürültü vektörü oluştur
                fake_data = generator(noise) # Sahte veri üret
                label.fill_(fake_label) # Etiketleri 'sahte' yap
                # Generator'ı eğitirken kullanacağımız için fake_data'yı detach ETMİYORUZ
                # Ancak Discriminator gradyanlarının Generator'a akmasını istemiyorsak detach() kullanırız.
                # Standart DCGAN'de D eğitimi sırasında detach edilir.
                output_fake = discriminator(fake_data.detach()).view(-1)
                errD_fake = criterion(output_fake, label)
                if torch.isnan(errD_fake): logger.warning(f"NaN detected in errD_fake at epoch {epoch+1}, iter {i}"); continue
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item() # Sahte verinin ortalama D skoru (G güncellenmeden ÖNCE)

                # Toplam Discriminator kaybı ve optimizer adımı
                errD = errD_real + errD_fake
                # İsteğe bağlı: Gradyanları kontrol et (NaN veya Inf var mı?)
                # for p in discriminator.parameters():
                #    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                #        logger.warning(f"NaN/Inf gradient detected in Discriminator at epoch {epoch+1}, iter {i}")
                optimizerD.step()

                # -----------------
                #  Generator Eğitimi
                # -----------------
                # Discriminator'ı kandırmaya çalış (sahte veriyi gerçek gibi göstermeye çalış)
                generator.zero_grad()
                label.fill_(real_label) # Generator için sahte verinin hedefi 'gerçek' etiketi
                # Discriminator'ın güncellenmiş parametreleriyle sahte veriyi tekrar değerlendir
                output_fake_for_G = discriminator(fake_data).view(-1) # detach() KULLANMA!
                errG = criterion(output_fake_for_G, label)
                if torch.isnan(errG): logger.warning(f"NaN detected in errG at epoch {epoch+1}, iter {i}"); continue
                errG.backward()
                D_G_z2 = output_fake_for_G.mean().item() # Sahte verinin ortalama D skoru (G güncellendikten SONRA)
                # İsteğe bağlı: Gradyanları kontrol et
                # for p in generator.parameters():
                #     if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                #         logger.warning(f"NaN/Inf gradient detected in Generator at epoch {epoch+1}, iter {i}")
                optimizerG.step()

                # Kayıpları ve ilerlemeyi kaydet
                epoch_D_loss_total += errD.item()
                epoch_G_loss_total += errG.item()
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # tqdm ilerleme çubuğunu güncelle
                pbar_epoch.set_postfix({
                    'D_loss': f"{errD.item():.4f}",
                    'G_loss': f"{errG.item():.4f}",
                    'D(x)': f"{D_x:.4f}", # Gerçek veri skoru (1'e yakın olmalı)
                    'D(G(z))': f"{D_G_z1:.4f} -> {D_G_z2:.4f}" # Sahte veri skoru (D için 0'a, G için 1'e yakın olmalı)
                })
                iters += 1

        # Epoch sonu logları
        avg_epoch_D_loss = epoch_D_loss_total / len(dataloader)
        avg_epoch_G_loss = epoch_G_loss_total / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s] tamamlandı => Avg D Loss: {avg_epoch_D_loss:.4f}, Avg G Loss: {avg_epoch_G_loss:.4f}")

        # Checkpoint kaydetme (save_gan_models kullanarak)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(save_dir, f"gan_checkpoint_epoch_{epoch+1}.pt")
            try:
                save_gan_models(
                    netG=generator,
                    netD=discriminator,
                    optimG=optimizerG,
                    optimD=optimizerD,
                    epoch=epoch + 1,
                    save_path=save_path,
                    G_losses=G_losses, # İsteğe bağlı: Kayıpları da kaydet
                    D_losses=D_losses
                )
            except Exception as e:
                logger.error(f"Checkpoint kaydedilirken hata oluştu: {e}")

    # Eğitim Sonu
    total_time = time.time() - start_time
    logger.info(f"GAN Eğitimi Tamamlandı. Toplam Süre: {total_time:.2f} saniye")

    # Son modelleri ayrı kaydet (checkpoint'ten farklı olarak sadece model state'leri)
    final_G_path = os.path.join(save_dir, "generator_final.pt")
    final_D_path = os.path.join(save_dir, "discriminator_final.pt")
    torch.save(generator.state_dict(), final_G_path)
    torch.save(discriminator.state_dict(), final_D_path)
    logger.info(f"Son Generator modeli kaydedildi: {final_G_path}")
    logger.info(f"Son Discriminator modeli kaydedildi: {final_D_path}")

    # Kayıp listelerini döndür (grafik çizdirmek için)
    return final_G_path, G_losses, D_losses