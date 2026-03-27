"""
Loader_Data.py
Tạo Dataset và DataLoader cho cả 2 loại model:
  - SeqDataset  : dùng cho KimCNN, BiLSTM, RCNN, Transformer
  - PhoDataset  : dùng cho PhoBERT
Import file này từ KimCNN.py hoặc các file kịch bản khác.
"""
import sys
import os

# ── Tự động thêm root vào sys.path ──────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../Processing
root_path   = os.path.dirname(current_dir)                  # .../VS code
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Import vocab và label_map từ nguồn DUY NHẤT
from Processing.Vocabulary_LabelMap import vocab, label_map, VOCAB_SIZE, NUM_CLASSES, DATA_PATH

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['text_clean'] = df['text_clean'].fillna("").astype(str)

# ── Chia Train / Val / Test  (80 / 10 / 10) ─────────────────────
train_val, test_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(
    train_val, test_size=0.111, random_state=42, stratify=train_val['label'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ════════════════════════════════════════════════════════════════
# SeqDataset — dùng cho các model không phải BERT
# ════════════════════════════════════════════════════════════════
MAX_LEN_SEQ = 300

def text_to_seq(text: str, vocab: dict, max_len: int) -> list:
    tokens = text.split()
    seq = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(seq) < max_len:
        seq += [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class SeqDataset(Dataset):
    def __init__(self, df, vocab, label_map, max_len=MAX_LEN_SEQ):
        self.seqs   = [text_to_seq(t, vocab, max_len) for t in df['text_clean']]
        self.labels = [label_map[int(l)] for l in df['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.seqs[idx],   dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ════════════════════════════════════════════════════════════════
# PhoDataset — dùng cho PhoBERT
# ════════════════════════════════════════════════════════════════
MAX_LEN_PHO = 128

class PhoDataset(Dataset):
    def __init__(self, df, tokenizer, label_map, max_len=MAX_LEN_PHO):
        self.texts     = df['text_clean'].values
        self.labels    = [label_map[int(l)] for l in df['label']]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Tạo DataLoader cho SeqDataset ────────────────────────────────
BATCH_SEQ = 64

trn_seq = DataLoader(SeqDataset(train_df, vocab, label_map), batch_size=BATCH_SEQ, shuffle=True)
val_seq = DataLoader(SeqDataset(val_df,   vocab, label_map), batch_size=BATCH_SEQ)
tst_seq = DataLoader(SeqDataset(test_df,  vocab, label_map), batch_size=BATCH_SEQ)

# ── Tạo DataLoader cho PhoDataset ────────────────────────────────
BATCH_PHO = 16

pho_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

trn_pho = DataLoader(PhoDataset(train_df, pho_tokenizer, label_map), batch_size=BATCH_PHO, shuffle=True)
val_pho = DataLoader(PhoDataset(val_df,   pho_tokenizer, label_map), batch_size=BATCH_PHO)
tst_pho = DataLoader(PhoDataset(test_df,  pho_tokenizer, label_map), batch_size=BATCH_PHO)

print(f"✅ DataLoader sẵn sàng | VOCAB_SIZE={VOCAB_SIZE} | NUM_CLASSES={NUM_CLASSES}")

# Export để robustness.py dùng
__all__ = [
    'trn_seq', 'val_seq', 'tst_seq',
    'trn_pho', 'val_pho', 'tst_pho',
    'test_df', 'pho_tokenizer',
    'SeqDataset', 'PhoDataset',
    'MAX_LEN_SEQ', 'MAX_LEN_PHO',
]