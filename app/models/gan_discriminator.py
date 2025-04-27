# app/models/gan_discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels, seq_length, num_pitches):
        """
        GAN Discriminator Modeli (DCGAN benzeri yapı).

        Args:
            input_channels (int): Giriş kanalı sayısı (piyano rulosu için genellikle 1).
            seq_length (int): Piyano rulosunun zaman adımı sayısı.
            num_pitches (int): Piyano rulosunun nota sayısı.
        """
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_pitches = num_pitches

        # Hesaplamalarda kullanılacak temel boyutlar
        ndf = 64 # Number of discriminator filters in the first conv layer

        self.main = nn.Sequential(
            # input: (batch, input_channels, seq_length, num_pitches)
            # Katman 1: input_channels -> ndf
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Boyut: (ndf) x (seq/2) x (pitch/2)

            # Katman 2: ndf -> ndf * 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Boyut: (ndf * 2) x (seq/4) x (pitch/4)

            # Katman 3: ndf * 2 -> ndf * 4
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
             # Boyut: (ndf * 4) x (seq/8) x (pitch/8)

            # Katman 4: ndf * 4 -> ndf * 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
             # Boyut: (ndf * 8) x (seq/16) x (pitch/16)

            # Katman 5: Sonuç -> 1 (Olasılık skoru)
            # Kernel boyutu, bir önceki katmanın çıktısını 1x1'e düşürecek şekilde ayarlanmalı
            nn.Conv2d(ndf * 8, 1, kernel_size=(seq_length // 16, num_pitches // 16), stride=1, padding=0, bias=False),
            nn.Sigmoid() # Çıkışı 0 ile 1 arasına sıkıştırır
            # Boyut: (1) x 1 x 1
        )

    def forward(self, input_pianoroll):
        # input_pianoroll: (batch_size, input_channels, seq_length, num_pitches)
        output = self.main(input_pianoroll)
        # Çıktıyı (batch_size,) şekline getir
        return output.view(-1, 1).squeeze(1)