import torch
import torch.nn as nn
from utils.positional_encoding import PositionalEncoding



# GPT benzeri model - daha güçlü bir seçenek
class GPTModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # GPT-style decoder block (unidirectional/casual attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Create a square attention mask to ensure autoregressive property
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1000, 1000) * float('-inf'), diagonal=1)
        )
        
    def forward(self, x):
        # x boyutu: (batch_size, seq_length)
        seq_length = x.size(1)
        
        embed = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embed = embed.transpose(0, 1)  # (seq_length, batch_size, embedding_dim)
        embed = self.positional_encoding(embed)
        
        # Apply attention mask
        mask = self.mask[:seq_length, :seq_length]
        
        # Use zeros as memory since we're using decoder-only architecture
        memory = torch.zeros_like(embed)
        
        output = self.transformer(embed, memory, tgt_mask=mask)
        output = output.transpose(0, 1)
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return output