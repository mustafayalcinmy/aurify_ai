# app/models/gpt.py
import torch
import torch.nn as nn
# PositionalEncoding'i doğru yoldan import et
from app.utils.positional_encoding import PositionalEncoding

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        """
        GPT Benzeri Model (Decoder-Only Transformer)

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embedding_dim (int): Gömme boyutu.
            num_heads (int): Çoklu dikkat (multi-head attention) sayısı.
            num_layers (int): Transformer decoder katman sayısı.
            dropout (float): Dropout oranı.
        """
        super(GPTModel, self).__init__()
        # Parametreleri doğrudan kullan
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4, # Genellikle embedding_dim'in 4 katı kullanılır
            dropout=dropout,
            batch_first=True # batch_first=True ekle
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

        # Dikkat maskesi (causal mask)
        # Maksimum sekans uzunluğuna göre oluşturmak daha esnek olabilir
        # Ancak şimdilik sabit bir boyutla bırakılabilir veya dinamik hale getirilebilir
        self.register_buffer("mask", None)


    def _generate_square_subsequent_mask(self, sz, device):
         mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
         return mask

    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        device = x.device

        # Maskeyi oluştur veya al
        if self.mask is None or self.mask.size(0) != seq_length:
             self.mask = self._generate_square_subsequent_mask(seq_length, device)


        embed = self.embedding(x) * (self.embedding_dim ** 0.5) # Scaling
        # embed: (batch_size, seq_length, embedding_dim)
        pos_encoded = self.positional_encoding(embed)
        # pos_encoded: (batch_size, seq_length, embedding_dim)

        # Decoder-only olduğu için memory (encoder çıktısı) kullanmıyoruz (None veya zero tensor)
        # PyTorch TransformerDecoder'ı encoder çıktısı olmadan kullanmak için
        # memory'yi target (yani pos_encoded) ile aynı yapabiliriz.
        output = self.transformer(pos_encoded, pos_encoded, tgt_mask=self.mask) # memory=pos_encoded
        # output: (batch_size, seq_length, embedding_dim)
        output = self.fc(output)
        # output: (batch_size, seq_length, vocab_size)
        return output