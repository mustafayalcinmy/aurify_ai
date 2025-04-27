import torch
from tqdm import tqdm


def generate_music(model, model_type, start_sequence, length=100, device='cpu', temperature=1.0):
    model.to(device)
    model.eval()
    generated = start_sequence.copy()
    input_seq = torch.tensor(start_sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad(), tqdm(total=length, desc="Müzik Üretiliyor", unit="nota") as pbar:
        hidden = None  # LSTM/GRU için başlangıç gizli durumu
        
        for _ in range(length):
            # Model tipine göre ileri geçiş
            if model_type in ['lstm', 'gru']:
                output, hidden = model(input_seq, hidden)
                next_token_logits = output[0, -1, :]
            else:
                if model_type == 'transformer':
                    output = model(input_seq)
                else:  # gpt
                    output = model(input_seq)
                next_token_logits = output[0, -1, :]
            
            # Sıcaklık uygulaması (temperature sampling)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            next_token_prob = torch.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(next_token_prob, 1).item()
            generated.append(next_token)
            
            # Sonraki girdi sekansını hazırla
            if model_type in ['lstm', 'gru']:
                input_seq = torch.tensor([[next_token]], dtype=torch.long, device=device)
            else:
                input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
                # Uzunluğu sınırla (bellek tasarrufu için)
                input_seq = input_seq[:, -len(start_sequence):]
            
            pbar.update(1)
            pbar.set_postfix({'son_üretilen': next_token})
    
    return generated