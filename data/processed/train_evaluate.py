"""
train_evaluate.py
Hàm train và đánh giá dùng chung cho tất cả các model dạng Seq (không phải BERT).
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ════════════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════════════
def evaluate_seq(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for texts, lbls in loader:
            texts, lbls = texts.to(device), lbls.to(device)
            outputs     = model(texts)
            loss        = criterion(outputs, lbls)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())

    avg_loss = total_loss / len(loader)
    return all_labels, all_preds, avg_loss


# ════════════════════════════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════════════════════════════
def train_seq_model(model, train_loader, val_loader, test_loader,
                    label_map, model_name, device,
                    epochs=12, lr=1e-3, patience=3, weight_decay=1e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc      = 0.0
    patience_counter  = 0

    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for texts, lbls in train_loader:
            texts, lbls = texts.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(texts), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────
        val_labels, val_preds, val_loss = evaluate_seq(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # ── Early Stopping ─────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping tại epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
                break

    # ── Test ───────────────────────────────────────────────────
    test_labels, test_preds, _ = evaluate_seq(model, test_loader, criterion, device)
    return model, history, test_preds, test_labels


# ════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════
def print_metrics(test_labels, test_preds, label_map, model_name):
    target_names = [str(k) for k in sorted(label_map.keys())]
    acc  = accuracy_score(test_labels, test_preds)
    mf1  = f1_score(test_labels, test_preds, average='macro')
    wf1  = f1_score(test_labels, test_preds, average='weighted')

    print(f"\n📊 [{model_name}]")
    print(f"  Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=target_names))
    return acc, mf1, wf1


# ════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════
def plot_results(history, test_labels, test_preds, label_map, title, cmap='Blues'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Learning curve
    axes[0].plot(history['train_loss'], label='Train Loss', color='red',  marker='o')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='blue', linestyle='--', marker='s')
    axes[0].set_title(f"{title} — Learning Curve")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Confusion matrix (normalized)
    cm  = confusion_matrix(test_labels, test_preds)
    cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    tick_labels = [str(k) for k in sorted(label_map.keys())]

    sns.heatmap(cmn, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=tick_labels, yticklabels=tick_labels,
                ax=axes[1])
    axes[1].set_title(f"{title} — Confusion Matrix (%)")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.show()
    plt.close()