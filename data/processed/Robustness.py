"""
robustness.py
Kiểm tra độ bền (Robustness) của model với 3 loại nhiễu:
  1. Typo         — hoán đổi 2 ký tự ngẫu nhiên trong từ
  2. Không dấu    — xóa toàn bộ dấu tiếng Việt
  3. Rút gọn      — xóa ngẫu nhiên 30% từ trong câu
Và đánh giá theo độ dài văn bản (ngắn / trung bình / dài).
"""
import sys, os
import random
import unicodedata
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Không cần GUI — lưu file PNG
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ── Thêm root vào sys.path ───────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(current_dir)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Processing.Loader_Data import SeqDataset, PhoDataset, MAX_LEN_SEQ, MAX_LEN_PHO

SAVE_DIR = os.path.join(root_path, "Results")
os.makedirs(SAVE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# HÀM THÊM NHIỄU
# ════════════════════════════════════════════════════════════════
def remove_accent(text: str) -> str:
    """Xóa toàn bộ dấu tiếng Việt."""
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized
                   if unicodedata.category(c) != 'Mn')

def add_typo(text: str, rate: float = 0.1) -> str:
    """Hoán đổi 2 ký tự liền kề trong ~rate% số từ."""
    words = text.split()
    noisy = []
    for w in words:
        if len(w) > 3 and random.random() < rate:
            i = random.randint(0, len(w) - 2)
            w = w[:i] + w[i+1] + w[i] + w[i+2:]
        noisy.append(w)
    return ' '.join(noisy)

def drop_words(text: str, rate: float = 0.3) -> str:
    """Xóa ngẫu nhiên rate% số từ."""
    words = text.split()
    kept  = [w for w in words if random.random() > rate]
    return ' '.join(kept) if kept else words[0]

NOISE_FUNCS = {
    'clean'      : lambda x: x,
    'typo'       : lambda x: add_typo(x, rate=0.1),
    'no_accent'  : remove_accent,
    'drop_words' : lambda x: drop_words(x, rate=0.3),
}

# ════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ THEO NOISE — SeqModel
# ════════════════════════════════════════════════════════════════
def eval_noise_seq(model, df, vocab, label_map, device, noise_name, noise_fn):
    df_noisy = df.copy()
    df_noisy['text_clean'] = df_noisy['text_clean'].apply(noise_fn)
    loader = DataLoader(
        SeqDataset(df_noisy, vocab, label_map, max_len=MAX_LEN_SEQ),
        batch_size=64
    )
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for texts, lbls in loader:
            texts, lbls = texts.to(device), lbls.to(device)
            out = model(texts)
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    acc = accuracy_score(labels, preds)
    print(f"  [{noise_name:12s}] Accuracy: {acc:.4f}")
    return acc

# ════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ THEO NOISE — PhoBERT
# ════════════════════════════════════════════════════════════════
def eval_noise_pho(model, df, tokenizer, label_map, device, noise_name, noise_fn):
    df_noisy = df.copy()
    df_noisy['text_clean'] = df_noisy['text_clean'].apply(noise_fn)
    loader = DataLoader(
        PhoDataset(df_noisy, tokenizer, label_map, max_len=MAX_LEN_PHO),
        batch_size=16
    )
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            out  = model(ids, mask)
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    acc = accuracy_score(labels, preds)
    print(f"  [{noise_name:12s}] Accuracy: {acc:.4f}")
    return acc

# ════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ THEO ĐỘ DÀI VĂN BẢN
# ════════════════════════════════════════════════════════════════
def eval_by_length_seq(model, df, vocab, label_map, device):
    df = df.copy()
    df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
    bins   = [0, 100, 300, 9999]
    labels_bin = ['Ngắn (<100)', 'Trung bình (100-300)', 'Dài (>300)']
    df['length_bin'] = pd.cut(df['word_count'], bins=bins, labels=labels_bin)

    results = {}
    for bin_name in labels_bin:
        sub = df[df['length_bin'] == bin_name]
        if len(sub) == 0:
            continue
        loader = DataLoader(
            SeqDataset(sub, vocab, label_map, max_len=MAX_LEN_SEQ),
            batch_size=64
        )
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for texts, lbls in loader:
                texts, lbls = texts.to(device), lbls.to(device)
                out = model(texts)
                preds.extend(torch.argmax(out, dim=1).cpu().tolist())
                true_labels.extend(lbls.cpu().tolist())
        acc = accuracy_score(true_labels, preds)
        results[bin_name] = acc
        print(f"  [{bin_name:25s}] n={len(sub):4d} | Accuracy: {acc:.4f}")
    return results

# ════════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ ROBUSTNESS
# ════════════════════════════════════════════════════════════════
def plot_robustness(noise_results, length_results, model_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Noise bar chart
    axes[0].bar(noise_results.keys(), noise_results.values(),
                color=['#2196F3','#FF9800','#F44336','#9C27B0'])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(f"{model_name} — Accuracy theo loại nhiễu")
    axes[0].set_ylabel("Accuracy")
    for i, (k, v) in enumerate(noise_results.items()):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

    # Length bar chart
    if length_results:
        axes[1].bar(length_results.keys(), length_results.values(),
                    color=['#4CAF50','#2196F3','#FF5722'])
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title(f"{model_name} — Accuracy theo độ dài văn bản")
        axes[1].set_ylabel("Accuracy")
        for i, (k, v) in enumerate(length_results.items()):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"robustness_{model_name.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Robustness chart saved: {save_path}")

# ════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ════════════════════════════════════════════════════════════════
def run_robustness_seq(model, test_df, vocab, label_map, device, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"  🔬 ROBUSTNESS TEST — {model_name}")
    print(f"{'='*60}")

    print("\n📌 Theo loại nhiễu:")
    noise_results = {}
    for name, fn in NOISE_FUNCS.items():
        noise_results[name] = eval_noise_seq(model, test_df, vocab, label_map, device, name, fn)

    print("\n📌 Theo độ dài văn bản:")
    length_results = eval_by_length_seq(model, test_df, vocab, label_map, device)

    plot_robustness(noise_results, length_results, model_name, SAVE_DIR)


def run_robustness_pho(model, test_df, tokenizer, label_map, device, model_name="PhoBERT"):
    print(f"\n{'='*60}")
    print(f"  🔬 ROBUSTNESS TEST — {model_name}")
    print(f"{'='*60}")

    print("\n📌 Theo loại nhiễu:")
    noise_results = {}
    for name, fn in NOISE_FUNCS.items():
        noise_results[name] = eval_noise_pho(model, test_df, tokenizer, label_map, device, name, fn)

    # PhoBERT dùng SeqDataset length bins (dùng chung df)
    print("\n📌 Theo độ dài văn bản:")
    # Reuse seq length eval với vocab dummy — chỉ đếm độ dài
    from Processing.Vocabulary_LabelMap import vocab
    length_results = {}
    df = test_df.copy()
    df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
    bins       = [0, 100, 300, 9999]
    bin_labels = ['Ngắn (<100)', 'Trung bình (100-300)', 'Dài (>300)']
    df['length_bin'] = pd.cut(df['word_count'], bins=bins, labels=bin_labels)

    for bin_name in bin_labels:
        sub = df[df['length_bin'] == bin_name]
        if len(sub) == 0:
            continue
        loader = DataLoader(
            PhoDataset(sub, tokenizer, label_map, max_len=MAX_LEN_PHO),
            batch_size=16
        )
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                out  = model(ids, mask)
                preds.extend(torch.argmax(out, dim=1).cpu().tolist())
                true_labels.extend(lbls.cpu().tolist())
        acc = accuracy_score(true_labels, preds)
        length_results[bin_name] = acc
        print(f"  [{bin_name:25s}] n={len(sub):4d} | Accuracy: {acc:.4f}")

    plot_robustness(noise_results, length_results, model_name, SAVE_DIR)