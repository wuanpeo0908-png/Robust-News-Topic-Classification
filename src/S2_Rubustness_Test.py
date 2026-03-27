"""
Robustness_S2.py — Robustness Test cho 5 model S2 (NamSyntax)
Data: s2_clean.csv (đã clean)
Đặt tại: VS code/
"""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = current_dir
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import random, unicodedata
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Processing.Vocabulary_LabelMap import vocab, label_map, VOCAB_SIZE, NUM_CLASSES
from Processing.Loader_Data         import SeqDataset, PhoDataset, MAX_LEN_SEQ, MAX_LEN_PHO, pho_tokenizer
from Processing.Structure_5models   import (KimCNN, BiLSTM_Attention, RCNN,
                                             TransformerClassifier, PhoBertClassifier)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(root_path, "Results", "Robustness_S2")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load S2 data ─────────────────────────────────────────────────
DATA_PATH = r"D:\Phenikaa-Study\kì 2 2025-2026\Xử lý ngôn ngữ\VS code\S2_Data\s2_clean.csv"
df = pd.read_csv(DATA_PATH)
df['text_clean'] = df['text_clean'].fillna('').astype(str)
df['label']      = df['label'].astype(int)

_, test_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['label'])
print(f"🚀 Device: {device} | S2 Test set: {len(test_df)} mẫu")

# ════════════════════════════════════════════════════════════════
# NOISE FUNCTIONS
# ════════════════════════════════════════════════════════════════
def remove_accent(text):
    nfd = unicodedata.normalize('NFD', str(text))
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def add_typo(text, rate=0.1):
    words = str(text).split()
    out = []
    for w in words:
        if len(w) > 3 and random.random() < rate:
            i = random.randint(0, len(w)-2)
            w = w[:i] + w[i+1] + w[i] + w[i+2:]
        out.append(w)
    return ' '.join(out)

def drop_words(text, rate=0.3):
    words = str(text).split()
    kept  = [w for w in words if random.random() > rate]
    return ' '.join(kept) if kept else words[0]

NOISE_FUNCS = {
    'clean'      : lambda x: x,
    'typo'       : lambda x: add_typo(x, 0.1),
    'no_accent'  : remove_accent,
    'drop_words' : lambda x: drop_words(x, 0.3),
}

# ════════════════════════════════════════════════════════════════
# EVAL FUNCTIONS
# ════════════════════════════════════════════════════════════════
def eval_noise_seq(model, df, noise_fn):
    df2 = df.copy()
    df2['text_clean'] = df2['text_clean'].apply(noise_fn)
    loader = DataLoader(SeqDataset(df2, vocab, label_map, MAX_LEN_SEQ), batch_size=64)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for texts, lbls in loader:
            texts, lbls = texts.to(device), lbls.to(device)
            preds.extend(torch.argmax(model(texts), 1).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    return accuracy_score(labels, preds)

def eval_noise_pho(model, df, noise_fn):
    df2 = df.copy()
    df2['text_clean'] = df2['text_clean'].apply(noise_fn)
    loader = DataLoader(PhoDataset(df2, pho_tokenizer, label_map, MAX_LEN_PHO), batch_size=16)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            preds.extend(torch.argmax(model(ids, mask), 1).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    return accuracy_score(labels, preds)

def eval_length(model, df, mtype):
    df2 = df.copy()
    df2['wc']  = df2['text_clean'].str.split().str.len()
    bins       = [0, 100, 300, 9999]
    bin_labels = ['Short(<100)', 'Medium(100-300)', 'Long(>300)']
    df2['bin'] = pd.cut(df2['wc'], bins=bins, labels=bin_labels)
    results = {}
    for b in bin_labels:
        sub = df2[df2['bin'] == b]
        if len(sub) == 0: continue
        if mtype == 'seq':
            loader = DataLoader(SeqDataset(sub, vocab, label_map, MAX_LEN_SEQ), batch_size=64)
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for texts, lbls in loader:
                    texts, lbls = texts.to(device), lbls.to(device)
                    preds.extend(torch.argmax(model(texts), 1).cpu().tolist())
                    labels.extend(lbls.cpu().tolist())
        else:
            loader = DataLoader(PhoDataset(sub, pho_tokenizer, label_map, MAX_LEN_PHO), batch_size=16)
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for batch in loader:
                    ids  = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    lbls = batch['labels'].to(device)
                    preds.extend(torch.argmax(model(ids, mask), 1).cpu().tolist())
                    labels.extend(lbls.cpu().tolist())
        acc = accuracy_score(labels, preds)
        results[b] = acc
        print(f"    [{b:20s}] n={len(sub):4d} | Acc: {acc:.4f}")
    return results

# ════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════
def plot_robustness(noise_res, length_res, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors  = ['#2196F3','#FF9800','#F44336','#9C27B0']
    colors2 = ['#4CAF50','#2196F3','#FF5722']

    bars = axes[0].bar(noise_res.keys(), noise_res.values(), color=colors)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(f"{model_name} — Accuracy by Noise Type")
    axes[0].set_ylabel("Accuracy")
    for bar, val in zip(bars, noise_res.values()):
        axes[0].text(bar.get_x()+bar.get_width()/2, val+0.01,
                     f"{val:.3f}", ha='center', fontsize=10)

    if length_res:
        bars2 = axes[1].bar(length_res.keys(), length_res.values(), color=colors2)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title(f"{model_name} — Accuracy by Text Length")
        axes[1].set_ylabel("Accuracy")
        for bar, val in zip(bars2, length_res.values()):
            axes[1].text(bar.get_x()+bar.get_width()/2, val+0.01,
                         f"{val:.3f}", ha='center', fontsize=10)

    plt.suptitle(f"S2 Robustness: {model_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"robustness_S2_{model_name.replace('+','').replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {path}")

# ════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ════════════════════════════════════════════════════════════════
MODELS = {
    "KimCNN"          : (KimCNN(VOCAB_SIZE, 128, 100, [3,4,5], NUM_CLASSES, 0.5),             "kimcnn_s2.pt",           "seq"),
    "BiLSTM_Attention": (BiLSTM_Attention(VOCAB_SIZE, 128, 128, NUM_CLASSES, 2, 0.5),         "bilstm_attention_s2.pt", "seq"),
    "RCNN"            : (RCNN(VOCAB_SIZE, 128, 256, NUM_CLASSES, dropout=0.5),                "rcnn_s2.pt",             "seq"),
    "Transformer"     : (TransformerClassifier(VOCAB_SIZE, 128, 8, 256, 2, NUM_CLASSES, 0.3), "transformer_s2.pt",      "seq"),
    "PhoBERT"         : (PhoBertClassifier(output_dim=NUM_CLASSES, dropout=0.1),              "phobert_s2.pt",          "pho"),
}

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
rows = []
for name, (model, weight_file, mtype) in MODELS.items():
    wpath = os.path.join(root_path, "Saved_Models", weight_file)
    if not os.path.exists(wpath):
        print(f"⚠️ Thiếu: {wpath} — bỏ qua")
        continue

    if mtype == 'pho':
        model = PhoBertClassifier(output_dim=NUM_CLASSES, dropout=0.1)
    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
    model.to(device)
    print(f"\n{'='*55}\n  🔬 S2 Robustness — {name}\n{'='*55}")

    noise_res = {}
    print("  📌 Noise test:")
    for noise_name, fn in NOISE_FUNCS.items():
        acc = eval_noise_seq(model, test_df, fn) if mtype == 'seq' \
              else eval_noise_pho(model, test_df, fn)
        noise_res[noise_name] = acc
        print(f"    [{noise_name:12s}] Acc: {acc:.4f}")

    print("  📌 Length test:")
    length_res = eval_length(model, test_df, mtype)

    plot_robustness(noise_res, length_res, name)
    rows.append({'Model': name,
                 **{f"noise_{k}": v for k, v in noise_res.items()},
                 **{f"len_{k}":   v for k, v in length_res.items()}})

csv_path = os.path.join(root_path, "Results", "robustness_s2_results.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\n✅ Xong! CSV: {csv_path}")
print(f"📁 Charts: {SAVE_DIR}/")
