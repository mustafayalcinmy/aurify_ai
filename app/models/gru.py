import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=64, hidden_dim=128, num_layers=1):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.gru(embed, hidden)
        output = self.fc(output)
        return output, hidden