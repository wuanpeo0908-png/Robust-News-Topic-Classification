"""
S2_3_Train.py — Bước 3: Train 5 model từ đầu trên NamSyntax (10 epochs)
Đọc s2_clean.csv → chia Train/Val/Test → train → lưu weights _s2.pt
Chạy SAU S2_2_Preprocess.py
"""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(os.path.dirname(current_dir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from Processing.Vocabulary_LabelMap import vocab, label_map, VOCAB_SIZE, NUM_CLASSES
from Processing.Loader_Data         import SeqDataset, PhoDataset, MAX_LEN_SEQ, MAX_LEN_PHO
from Processing.Structure_5models   import (KimCNN, BiLSTM_Attention, RCNN,
                                             TransformerClassifier, PhoBertClassifier)
from Processing.train_evaluate      import train_seq_model, print_metrics, plot_results
from Processing.Loader_Data         import pho_tokenizer

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(root_path, "S2_Data")
SAVE_DIR = os.path.join(root_path, "Saved_Models")
RES_DIR  = os.path.join(root_path, "Results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RES_DIR,  exist_ok=True)

CLEAN_PATH = os.path.join(DATA_DIR, "s2_clean.csv")

print(f"🚀 Device: {device}")

if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy {CLEAN_PATH}\n→ Chạy S2_2_Preprocess.py trước!")

# ════════════════════════════════════════════════════════════════
# 1. LOAD & CHIA DATA
# ════════════════════════════════════════════════════════════════
df = pd.read_csv(CLEAN_PATH)
df['label']      = df['label'].astype(int)
df['text_clean'] = df['text_clean'].fillna('').astype(str)
print(f"✅ Đã load: {len(df)} mẫu | Nhãn: {sorted(df['label'].unique())}")

# Chia Train 80% / Val 10% / Test 10% — stratified
train_val, test_df  = train_test_split(df,       test_size=0.10,  random_state=42, stratify=df['label'])
train_df,  val_df   = train_test_split(train_val, test_size=0.111, random_state=42, stratify=train_val['label'])

print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── DataLoader Seq ────────────────────────────────────────────────
trn_seq = DataLoader(SeqDataset(train_df, vocab, label_map, MAX_LEN_SEQ), batch_size=64, shuffle=True)
val_seq = DataLoader(SeqDataset(val_df,   vocab, label_map, MAX_LEN_SEQ), batch_size=64)
tst_seq = DataLoader(SeqDataset(test_df,  vocab, label_map, MAX_LEN_SEQ), batch_size=64)

# ── DataLoader PhoBERT ────────────────────────────────────────────
trn_pho = DataLoader(PhoDataset(train_df, pho_tokenizer, label_map, MAX_LEN_PHO), batch_size=16, shuffle=True)
val_pho = DataLoader(PhoDataset(val_df,   pho_tokenizer, label_map, MAX_LEN_PHO), batch_size=16)
tst_pho = DataLoader(PhoDataset(test_df,  pho_tokenizer, label_map, MAX_LEN_PHO), batch_size=16)

# ════════════════════════════════════════════════════════════════
# 2. CẤU HÌNH 5 MODEL — giống hệt S1
# ════════════════════════════════════════════════════════════════
EPOCHS = 10

seq_models = {
    "KimCNN": (
        KimCNN(vocab_size=VOCAB_SIZE, embed_dim=128, n_filters=100,
               filter_sizes=[3,4,5], output_dim=NUM_CLASSES, dropout=0.5),
        "kimcnn_s2.pt", "Blues"
    ),
    "BiLSTM+Attention": (
        BiLSTM_Attention(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=128,
                         output_dim=NUM_CLASSES, n_layers=2, dropout=0.5),
        "bilstm_attention_s2.pt", "Greens"
    ),
    "RCNN": (
        RCNN(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
             output_dim=NUM_CLASSES, dropout=0.5),
        "rcnn_s2.pt", "Oranges"
    ),
    "Transformer": (
        TransformerClassifier(vocab_size=VOCAB_SIZE, embed_dim=128, nhead=8,
                              nhid=256, nlayers=2, output_dim=NUM_CLASSES, dropout=0.3),
        "transformer_s2.pt", "Purples"
    ),
}

# ════════════════════════════════════════════════════════════════
# 3. HÀM TRAIN/EVAL PHOBERT
# ════════════════════════════════════════════════════════════════
import torch.nn as nn

def train_pho(model, trn, val, tst, epochs=EPOCHS, lr=2e-5, patience=3):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    history   = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, patience_counter = 0.0, 0

    print(f"\n{'='*60}\n  S2 — PhoBERT\n{'='*60}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in trn:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(trn)

        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                out  = model(ids, mask)
                val_loss_total += criterion(out, lbls).item()
                val_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
                val_labels_list.extend(lbls.cpu().tolist())

        val_loss = val_loss_total / len(val)
        val_acc  = accuracy_score(val_labels_list, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping (best: {best_val_acc:.4f})")
                break

    # Test
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tst:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            out  = model(ids, mask)
            test_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            test_labels.extend(lbls.cpu().tolist())

    return model, history, test_preds, test_labels

# ════════════════════════════════════════════════════════════════
# 4. TRAIN TẤT CẢ MODEL
# ════════════════════════════════════════════════════════════════
all_results = {}

# ── 4a. Seq Models ───────────────────────────────────────────────
for name, (model, save_name, cmap) in seq_models.items():
    model = model.to(device)
    print(f"\n{'='*60}\n  S2 — {name}\n{'='*60}")
    print(f"  Tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, history, test_preds, test_labels = train_seq_model(
        model, trn_seq, val_seq, tst_seq,
        label_map, f"S2 — {name}", device,
        epochs=EPOCHS, lr=1e-3, patience=3
    )

    acc, mf1, wf1 = print_metrics(test_labels, test_preds, label_map, f"S2 {name}")
    all_results[name] = {'Accuracy': acc, 'Macro-F1': mf1, 'Weighted-F1': wf1}

    plot_results(history, test_labels, test_preds, label_map, f"S2: {name}", cmap=cmap)

    # Lưu weights
    save_path = os.path.join(SAVE_DIR, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"  💾 Đã lưu: {save_path}")

# ── 4b. PhoBERT ──────────────────────────────────────────────────
# Freeze toàn bộ, chỉ unfreeze 2 layer cuối
pho_model = PhoBertClassifier(output_dim=NUM_CLASSES, dropout=0.1).to(device)
for param in pho_model.phobert.parameters():
    param.requires_grad = False
for name_p, param in pho_model.phobert.named_parameters():
    if "encoder.layer.11" in name_p or "encoder.layer.10" in name_p:
        param.requires_grad = True

pho_model, history, test_preds, test_labels = train_pho(
    pho_model, trn_pho, val_pho, tst_pho,
    epochs=EPOCHS, lr=2e-5, patience=3
)

target_names = [str(k) for k in sorted(label_map.keys())]
acc  = accuracy_score(test_labels, test_preds)
mf1  = f1_score(test_labels, test_preds, average='macro')
wf1  = f1_score(test_labels, test_preds, average='weighted')
print(f"\n📊 [S2 PhoBERT] Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}")
print(classification_report(test_labels, test_preds, target_names=target_names))
all_results['PhoBERT'] = {'Accuracy': acc, 'Macro-F1': mf1, 'Weighted-F1': wf1}

plot_results(history, test_labels, test_preds, label_map, "S2: PhoBERT", cmap='Reds')

pho_save = os.path.join(SAVE_DIR, "phobert_s2.pt")
torch.save(pho_model.state_dict(), pho_save)
print(f"  💾 Đã lưu: {pho_save}")

# ════════════════════════════════════════════════════════════════
# 5. TỔNG HỢP KẾT QUẢ S2
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  📊 TỔNG HỢP KẾT QUẢ S2 — Train trên NamSyntax")
print(f"{'='*60}")
results_df = pd.DataFrame(all_results).T.sort_values('Accuracy', ascending=False)
print(results_df.round(4).to_string())

csv_path = os.path.join(RES_DIR, "s2_train_results.csv")
results_df.to_csv(csv_path)
print(f"\n✅ Đã lưu kết quả: {csv_path}")

# Biểu đồ so sánh
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['Accuracy', 'Macro-F1', 'Weighted-F1']
colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
names   = list(all_results.keys())

for i, metric in enumerate(metrics):
    vals = [all_results[m][metric] for m in names]
    bars = axes[i].bar(names, vals, color=colors[:len(names)])
    axes[i].set_ylim(0, 1.0)
    axes[i].set_title(f"S2 — {metric}")
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     val + 0.01, f"{val:.3f}", ha='center', fontsize=9)

plt.suptitle("Scenario 2: Train trên NamSyntax (10 epochs)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
chart_path = os.path.join(RES_DIR, "s2_train_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"📈 Biểu đồ: {chart_path}")
print("\n→ Chạy tiếp: S3 (Input=VnExpress, Test=NamSyntax)")