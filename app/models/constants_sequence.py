# app/models/constants_sequence.py

# Ortak Parametreler (Varsa)
SEQUENCE_COMMON = {
    "vocab_size": 128,
}

# LSTM Modeli Parametreleri
LSTM_PARAMS = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    **SEQUENCE_COMMON # Ortak parametreleri ekle
}

# GRU Modeli Parametreleri
GRU_PARAMS = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    **SEQUENCE_COMMON
}

# Transformer Modeli Parametreleri
TRANSFORMER_PARAMS = {
    "embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    **SEQUENCE_COMMON
}

# GPT Modeli Parametreleri
GPT_PARAMS = {
    "embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    **SEQUENCE_COMMON
}

# Model tipi ismine göre parametreleri döndüren bir sözlük
SEQUENCE_MODEL_PARAMS = {
    "lstm": LSTM_PARAMS,
    "gru": GRU_PARAMS,
    "transformer": TRANSFORMER_PARAMS,
    "gpt": GPT_PARAMS,
}