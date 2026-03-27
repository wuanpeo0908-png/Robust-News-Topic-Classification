"""
PhoBERT.py  —  S5: PhoBERT + Custom Head
"""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(os.path.dirname(current_dir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from Processing.Vocabulary_LabelMap import label_map, NUM_CLASSES
from Processing.Loader_Data         import trn_pho, val_pho, tst_pho, test_df, pho_tokenizer
from Processing.Structure_5models   import PhoBertClassifier
from Processing.train_evaluate      import plot_results
from Processing.Robustness          import run_robustness_pho

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ── Khởi tạo model ───────────────────────────────────────────────
model = PhoBertClassifier(output_dim=NUM_CLASSES, dropout=0.1).to(device)

# Chỉ fine-tune 2 layer cuối của PhoBERT + classifier head để tiết kiệm VRAM
for name, param in model.phobert.named_parameters():
    param.requires_grad = False                        # Freeze toàn bộ trước
for name, param in model.phobert.named_parameters():
    if "encoder.layer.11" in name or "encoder.layer.10" in name or "pooler" in name:
        param.requires_grad = True                     # Unfreeze 2 layer cuối

total   = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Tổng tham số: {total:,} | Trainable: {trainable:,}")

# ── Hàm train/eval riêng cho PhoBERT (input khác SeqDataset) ─────
def evaluate_pho(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['labels'].to(device)
            out   = model(ids, mask)
            loss  = criterion(out, lbls)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    return all_labels, all_preds, total_loss / len(loader)

def train_pho_model(model, trn, val, tst,
                    epochs=6, lr=2e-5, patience=2):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, patience_counter = 0.0, 0

    print(f"\n{'='*60}\n  S5 — PhoBERT + Custom Head\n{'='*60}")

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
        val_labels, val_preds, val_loss = evaluate_pho(model, val, criterion)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping (best val acc: {best_val_acc:.4f})")
                break

    test_labels, test_preds, _ = evaluate_pho(model, tst, criterion)
    return model, history, test_preds, test_labels

# ── Train ─────────────────────────────────────────────────────────
model, history, test_preds, test_labels = train_pho_model(
    model, trn_pho, val_pho, tst_pho,
    epochs=6, lr=2e-5, patience=2
)

# ── Kết quả ───────────────────────────────────────────────────────
target_names = [str(k) for k in sorted(label_map.keys())]
acc  = accuracy_score(test_labels, test_preds)
mf1  = f1_score(test_labels, test_preds, average='macro')
wf1  = f1_score(test_labels, test_preds, average='weighted')
print(f"\n📊 [S5 PhoBERT]")
print(f"  Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}")
print(classification_report(test_labels, test_preds, target_names=target_names))

plot_results(history, test_labels, test_preds, label_map,
             "S5: PhoBERT", cmap='Reds')

# ── Robustness Test ───────────────────────────────────────────────
run_robustness_pho(model, test_df, pho_tokenizer, label_map, device, model_name="PhoBERT")
save_dir = os.path.join(root_path, "Saved_Models")
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "phobert_s1.pt"))
print("✅ Saved: phobert_s1.pt")