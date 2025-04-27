# app/models/gru.py
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        GRU Modeli

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embedding_dim (int): Gömme (embedding) boyutu.
            hidden_dim (int): GRU gizli katman boyutu.
            num_layers (int): GRU katman sayısı.
        """
        super(GRUModel, self).__init__()
        # Parametreleri doğrudan kullan
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.gru(embed, hidden)
        output = self.fc(output)
        return output, hidden