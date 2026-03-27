"""
Vocabulary_LabelMap.py
Tạo vocab và label_map từ data_ready_for_model.csv
Chạy độc lập hoặc được import bởi Loader_Data.py
"""
import sys
import os
import pandas as pd
from collections import Counter

# ── Tự động thêm root vào sys.path ──────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../Processing
root_path   = os.path.dirname(current_dir)                  # .../VS code
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# ── Đường dẫn data ───────────────────────────────────────────────
DATA_PATH = os.path.join(root_path, "Processing", "data_ready_for_model.csv")

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['text_clean'] = df['text_clean'].fillna("").astype(str)

# ── Tạo Vocab ────────────────────────────────────────────────────
MIN_FREQ = 2

all_words = []
for text in df['text_clean']:
    all_words.extend(text.split())

word_counts = Counter(all_words)

# index 0 = PAD, index 1 = UNK, từ thật bắt đầu từ index 2
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counts.items():
    if count >= MIN_FREQ:
        vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)

# ── Tạo Label Map ────────────────────────────────────────────────
sorted_labels = sorted(df['label'].unique())
label_map = {int(lbl): idx for idx, lbl in enumerate(sorted_labels)}
NUM_CLASSES = len(label_map)

# ── In thông tin khi chạy trực tiếp ─────────────────────────────
if __name__ == "__main__":
    print(f"✅ Vocab Size  : {VOCAB_SIZE}")
    print(f"✅ Num Classes : {NUM_CLASSES}")
    print(f"✅ Label Map   : {label_map}")