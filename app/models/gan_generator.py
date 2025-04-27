# app/models/gan_generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels, seq_length, num_pitches):
        """
        GAN Generator Modeli (DCGAN benzeri yapı).

        Args:
            latent_dim (int): Gürültü vektörü boyutu (örn: 100).
            output_channels (int): Çıkış kanalı sayısı (piyano rulosu için genellikle 1).
            seq_length (int): Piyano rulosunun zaman adımı sayısı (örn: 128).
            num_pitches (int): Piyano rulosunun nota sayısı (örn: 84).
        """
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.num_pitches = num_pitches
        self.output_channels = output_channels
        self.latent_dim = latent_dim

        # Hesaplamalarda kullanılacak temel boyutlar (DCGAN makalesindeki gibi)
        # Bu değerleri modelinizin derinliğine ve girdi/çıktı boyutlarına göre ayarlamanız gerekebilir.
        ngf = 64 # Number of generator filters in the last conv layer

        # Başlangıç boyutu hesaplaması (genellikle kernel_size, stride, padding'e göre ayarlanır)
        # Hedef: (latent_dim) x 1 x 1 -> (ngf*8) x (seq_length/8) x (num_pitches/8) gibi bir başlangıç
        # Basitlik için ConvTranspose2d'nin ilk katmanıyla başlıyoruz,
        # Girdiyi (batch, latent_dim, 1, 1) varsayıyoruz.

        # Katmanları tanımla
        self.main = nn.Sequential(
            # input: Z (batch, latent_dim, 1, 1)
            # nn.Linear + Reshape veya ConvTranspose2d kullanılabilir
            # Örnek: ConvTranspose2d ile başlama
            # Kernel size'ları seq_length ve num_pitches'e bağlı olarak dikkatli seçmek gerekir.
            # Bu örnekte seq=128, pitch=84 varsayımıyla kaba bir ayarlama yapılmıştır.
            # Daha sağlam bir yol, hedef boyutları belirleyip geriye doğru hesaplamaktır.

            # Katman 1: latent_dim -> ngf * 8
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=(seq_length // 16, num_pitches // 16), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Boyut: (ngf * 8) x (seq/16) x (pitch/16)

            # Katman 2: ngf * 8 -> ngf * 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False), # Boyutu 2 katına çıkarır
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
             # Boyut: (ngf * 4) x (seq/8) x (pitch/8)

            # Katman 3: ngf * 4 -> ngf * 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False), # Boyutu 2 katına çıkarır
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
             # Boyut: (ngf * 2) x (seq/4) x (pitch/4)

            # Katman 4: ngf * 2 -> ngf
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False), # Boyutu 2 katına çıkarır
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
             # Boyut: (ngf) x (seq/2) x (pitch/2)

            # Katman 5: ngf -> output_channels (Hedef boyuta ulaş)
            nn.ConvTranspose2d(ngf, output_channels, kernel_size=4, stride=2, padding=1, bias=False), # Boyutu 2 katına çıkarır
            nn.Tanh() # Çıkışı -1 ile 1 arasına sıkıştırır (Veri normalizasyonuna bağlı)
            # Veya nn.Sigmoid() # Çıkışı 0 ile 1 arasına sıkıştırır
            # Boyut: (output_channels) x seq_length x num_pitches
        )

    def forward(self, input_noise):
        # Gürültü vektörünü doğru şekle getir (batch_size, latent_dim, 1, 1)
        input_reshaped = input_noise.view(input_noise.size(0), self.latent_dim, 1, 1)
        output = self.main(input_reshaped)
        return output # Shape: (batch_size, output_channels, seq_length, num_pitches)