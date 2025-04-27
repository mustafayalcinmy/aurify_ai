
import torch.nn as nn
from utils.positional_encoding import PositionalEncoding


# Transformer tabanlı müzik üretim modeli
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        vocab_size: MIDI pitch dağarcığı boyutu.
        embedding_dim: Gömme boyutu.
        num_heads: Çoklu dikkat (multi-head attention) sayısı.
        num_layers: Transformer encoder katman sayısı.
        dropout: Dropout oranı.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        embed = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embed = embed.transpose(0, 1)  # (seq_length, batch_size, embedding_dim)
        embed = self.positional_encoding(embed)
        transformer_output = self.transformer(embed)  # (seq_length, batch_size, embedding_dim)
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, seq_length, embedding_dim)
        output = self.fc(transformer_output)  # (batch_size, seq_length, vocab_size)
        return output