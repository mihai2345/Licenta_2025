import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder (Feature Map → Proiecție la Embed Size)

class EncoderCNN_v2(nn.Module):
    # Acest encoder proiectează fiecare dintre cele 196 regiuni CNN (2048-dim) la embed_size
    def __init__(self, embed_size):
        super(EncoderCNN_v2, self).__init__()
        # 2048 este dimensiunea caracteristicilor ResNet/VGG pentru fiecare regiune
        self.fc = nn.Linear(2048, embed_size) 
        self.relu = nn.ReLU()
        
    def forward(self, features):
        # features este de forma [batch_size, num_pixels, 2048]
        # Aplica layer-ul fc pe ultima dimensiune (2048)
        encoded_features = self.fc(features)
        return self.relu(encoded_features)
        # Ieșire: [batch_size, num_pixels, embed_size]


#  Attention Mechanism

class Attention_v2(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention_v2, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Proiectează feature map-ul (V)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Proiectează starea ascunsă (H)
        self.full_att = nn.Linear(attention_dim, 1)              # Calculează scorul de energie (e)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)                         # Calculează ponderea (alpha)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch_size, num_pixels, encoder_dim] (V)
        # decoder_hidden: [batch_size, decoder_dim] (H)

        # 1. Proiecții
        att1 = self.encoder_att(encoder_out) # [B, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1) # [B, 1, attention_dim]

        # 2. Scorul de Energie (e): e = tanh(V*Wv + H*Wh)
        att = self.relu(att1 + att2) # [B, num_pixels, attention_dim]

        # 3. Scorul final de atenție
        e = self.full_att(att).squeeze(2) # [B, num_pixels]

        # 4. Alpha (ponderea): softmax(e)
        alpha = self.softmax(e) # [B, num_pixels]

        # 5. Context Vector (C): suma ponderată C = sum(alpha * V)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # [B, encoder_dim]

        return attention_weighted_encoding, alpha


#  Decoder (LSTM + Atenție)

class DecoderRNN_v2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0.5):
        super(DecoderRNN_v2, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Dimensiunile pentru Atenție
        attention_dim = embed_size # Poate fi diferit, dar e comun să fie egal cu embed_size
        self.attention = Attention_v2(encoder_dim=embed_size, 
                                      decoder_dim=hidden_size, 
                                      attention_dim=attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Input LSTM: embed_size (cuvânt) + embed_size (vectorul de Context din Atenție)
        self.lstm = nn.LSTMCell(embed_size + embed_size, hidden_size)
        
        # Layer-e de output
        self.f_beta = nn.Linear(hidden_size, embed_size) # Gating mechanism (Optional)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_out, captions):
        # encoder_out: [batch_size, num_pixels, embed_size] (Harta de feature-uri)
        
        batch_size = encoder_out.size(0)
        embeddings = self.embedding(captions)
        
        # Stări inițiale LSTM
        h, c = self.init_hidden_state(batch_size, encoder_out.device)
        
        outputs = torch.zeros(batch_size, captions.size(1), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, captions.size(1), encoder_out.size(1)).to(encoder_out.device)
        
        for t in range(captions.size(1)):
            # 1. Atenție
            context_vector, alpha = self.attention(encoder_out, h)
            
            # 2. Input LSTM: concatenare cuvânt (embeddings) și vector de context
            lstm_input = torch.cat([embeddings[:, t, :], context_vector], dim=1)
            
            # 3. LSTM
            h, c = self.lstm(lstm_input, (h, c))
            
            # 4. Gating (f_beta): opțional, ajută la ignorarea contextului
            gate = self.sigmoid(self.f_beta(h))
            weighted_h = gate * h 
            
            # 5. Output
            output = self.fc(self.dropout(weighted_h))
            
            outputs[:, t, :] = output
            alphas[:, t, :] = alpha
            
        return outputs, alphas

    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c


# Model combinat

class ImageCaptioningModel_v2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel_v2, self).__init__()
        self.encoder = EncoderCNN_v2(embed_size)
        self.decoder = DecoderRNN_v2(embed_size, hidden_size, vocab_size)

    def forward(self, features, captions):
        enc_out = self.encoder(features) # [B, Num_Pixels, Embed_Size]
        outputs, alphas = self.decoder(enc_out, captions)
        # Modelul de antrenament returnează output-ul și alpha (pentru loss/vizualizare)
        return outputs, alphas