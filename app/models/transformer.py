# app/models/transformer.py
import torch.nn as nn
# PositionalEncoding'i doğru yoldan import et (eğer utils app içindeyse)
from app.utils.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        """
        Transformer Modeli

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embedding_dim (int): Gömme boyutu.
            num_heads (int): Çoklu dikkat (multi-head attention) sayısı.
            num_layers (int): Transformer encoder katman sayısı.
            dropout (float): Dropout oranı.
        """
        super(TransformerModel, self).__init__()
        # Parametreleri doğrudan kullan
        self.embedding_dim = embedding_dim # positional encoding için sakla
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        # d_model embedding_dim ile aynı olmalı
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True) # batch_first=True ekle
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        embed = self.embedding(x) * (self.embedding_dim ** 0.5) # Scaling
        # embed: (batch_size, seq_length, embedding_dim)
        pos_encoded = self.positional_encoding(embed)
        # pos_encoded: (batch_size, seq_length, embedding_dim)
        transformer_output = self.transformer(pos_encoded)
        # transformer_output: (batch_size, seq_length, embedding_dim)
        output = self.fc(transformer_output)
        # output: (batch_size, seq_length, vocab_size)
        return output