"""
Structure_5models.py
Định nghĩa 5 kiến trúc mô hình:
  1. KimCNN
  2. BiLSTM_Attention
  3. RCNN
  4. TransformerClassifier
  5. PhoBertClassifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel


# ════════════════════════════════════════════════════════════════
# 1. KIM CNN
# ════════════════════════════════════════════════════════════════
class KimCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.fc      = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch, seq_len]
        embedded = self.embedding(text).unsqueeze(1)          # [batch, 1, seq_len, embed]
        conved   = [F.relu(conv(embedded)).squeeze(3)         # [batch, n_filters, seq_len-fs+1]
                    for conv in self.convs]
        pooled   = [F.max_pool1d(c, c.shape[2]).squeeze(2)   # [batch, n_filters]
                    for c in conved]
        cat      = self.dropout(torch.cat(pooled, dim=1))     # [batch, n_filters * len(filter_sizes)]
        return self.fc(cat)


# ════════════════════════════════════════════════════════════════
# 2. BiLSTM + ATTENTION
# ════════════════════════════════════════════════════════════════
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding        = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm             = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                                        bidirectional=True, batch_first=True, dropout=dropout)
        self.attention_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v                = nn.Parameter(torch.rand(hidden_dim * 2))
        self.fc               = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout          = nn.Dropout(dropout)

    def attention_net(self, lstm_output):
        energy  = torch.tanh(self.attention_linear(lstm_output))       # [batch, seq, hid*2]
        v       = self.v.unsqueeze(0).unsqueeze(2).expand(
                      lstm_output.size(0), -1, -1)                     # [batch, hid*2, 1]
        weights = torch.bmm(energy, v).squeeze(2)                      # [batch, seq]
        alphas  = F.softmax(weights, dim=1).unsqueeze(-1)              # [batch, seq, 1]
        return torch.sum(lstm_output * alphas, dim=1)                  # [batch, hid*2]

    def forward(self, text):
        embedded   = self.dropout(self.embedding(text))
        output, _  = self.lstm(embedded)
        attn_out   = self.attention_net(output)
        return self.fc(self.dropout(attn_out))


# ════════════════════════════════════════════════════════════════
# 3. RCNN
# ════════════════════════════════════════════════════════════════
class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1       = nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim * 2)
        self.fc2       = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, text):
        embedded    = self.embedding(text)                         # [batch, seq, embed]
        lstm_out, _ = self.lstm(embedded)                          # [batch, seq, hid*2]
        combined    = torch.cat((lstm_out, embedded), dim=2)       # [batch, seq, hid*2+embed]
        y2          = torch.tanh(self.fc1(combined)).permute(0,2,1)# [batch, hid*2, seq]
        y3          = F.max_pool1d(y2, y2.size(2)).squeeze(2)      # [batch, hid*2]
        return self.fc2(self.dropout(y3))


# ════════════════════════════════════════════════════════════════
# 4. TRANSFORMER CLASSIFIER
# ════════════════════════════════════════════════════════════════
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, output_dim, dropout=0.5):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder     = nn.Parameter(torch.zeros(1, 512, embed_dim))
        encoder_layer        = nn.TransformerEncoderLayer(embed_dim, nhead, nhid,
                                                          dropout, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc              = nn.Linear(embed_dim, output_dim)
        self.dropout         = nn.Dropout(dropout)

    def forward(self, text):
        embedded  = self.embedding(text) * math.sqrt(self.embedding.embedding_dim)
        embedded += self.pos_encoder[:, :text.size(1), :]
        output    = self.transformer_enc(embedded)
        output    = output.mean(dim=1)                 # mean pooling
        return self.fc(self.dropout(output))


# ════════════════════════════════════════════════════════════════
# 5. PHOBERT CLASSIFIER
# ════════════════════════════════════════════════════════════════
class PhoBertClassifier(nn.Module):
    def __init__(self, output_dim, dropout=0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base",use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs    = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.fc(self.dropout(cls_output))


print("✅ Structure_5models defined successfully!")