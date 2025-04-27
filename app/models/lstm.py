
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=64, hidden_dim=128, num_layers=1):
        """
        vocab_size: MIDI pitch değerleri (0-127) için kelime dağarcığı boyutu.
        embedding_dim: Gömme (embedding) boyutu.
        hidden_dim: LSTM gizli katman boyutu.
        num_layers: LSTM katman sayısı.
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden