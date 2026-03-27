"""
BiLSTM_Attention.py  —  S2: BiLSTM + Attention
"""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(os.path.dirname(current_dir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import numpy as np
from Processing.Vocabulary_LabelMap import vocab, label_map, VOCAB_SIZE, NUM_CLASSES
from Processing.Loader_Data         import trn_seq, val_seq, tst_seq, test_df
from Processing.Structure_5models   import BiLSTM_Attention
from Processing.train_evaluate      import train_seq_model, print_metrics, plot_results
from Processing.Robustness         import run_robustness_seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ── Khởi tạo model ───────────────────────────────────────────────
model = BiLSTM_Attention(
    vocab_size  = VOCAB_SIZE,
    embed_dim   = 128,
    hidden_dim  = 128,
    output_dim  = NUM_CLASSES,
    n_layers    = 2,
    dropout     = 0.5
).to(device)

print(f"  Tổng tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Train ─────────────────────────────────────────────────────────
model, history, test_preds, test_labels = train_seq_model(
    model, trn_seq, val_seq, tst_seq,
    label_map, "S2 — BiLSTM + Attention", device,
    epochs=12, lr=5e-4, patience=3
)

# ── Kết quả ───────────────────────────────────────────────────────
print_metrics(test_labels, test_preds, label_map, "S2 BiLSTM+Attention")
plot_results(history, test_labels, test_preds, label_map,
             "S2: BiLSTM+Attention", cmap='Greens')

# ── Robustness Test ───────────────────────────────────────────────
run_robustness_seq(model, test_df, vocab, label_map, device, model_name="BiLSTM+Attention")
save_dir = os.path.join(root_path, "Saved_Models")
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "bilstm_attention_s1.pt"))
print("✅ Saved: bilstm_attention_s1.pt")