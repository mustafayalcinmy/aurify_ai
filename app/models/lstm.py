# app/models/lstm.py
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        LSTM Modeli

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embedding_dim (int): Gömme (embedding) boyutu.
            hidden_dim (int): LSTM gizli katman boyutu.
            num_layers (int): LSTM katman sayısı.
        """
        super(LSTMModel, self).__init__()
        # Parametreleri doğrudan kullan
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden