"""
KimCNN.py  —  S1: Baseline
Huấn luyện mô hình KimCNN phân loại tin tức tiếng Việt (10 nhãn).
"""
import os
import sys
from xml.parsers.expat import model

# ── PHẢI ĐẶT TRƯỚC MỌI IMPORT TORCH ────────────────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # Lỗi CUDA báo đúng dòng

# ── Thêm root vào sys.path ───────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))           # .../S1_Baseline
scenarios   = os.path.dirname(current_dir)                         # .../Các Kịch Bản
root_path   = os.path.dirname(scenarios)                           # .../VS code
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from Processing.Vocabulary_LabelMap import vocab, label_map, VOCAB_SIZE, NUM_CLASSES
from Processing.Loader_Data         import trn_seq, val_seq, tst_seq
from Processing.Structure_5models   import KimCNN
from Processing.train_evaluate      import train_seq_model, print_metrics, plot_results
from Processing.Robustness          import run_robustness_seq

# ── Device ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ── Kiểm tra vocab trước khi train ──────────────────────────────
max_idx = max(max(seq) for seq in trn_seq.dataset.seqs)
print(f"  Max token index trong train : {max_idx}")
print(f"  VOCAB_SIZE                  : {VOCAB_SIZE}")
assert max_idx < VOCAB_SIZE, \
    f"❌ VOCAB overflow! Token index {max_idx} >= VOCAB_SIZE {VOCAB_SIZE}"
print("  ✅ Vocab OK")

# ── Cấu hình KimCNN ──────────────────────────────────────────────
model_kim = KimCNN(
    vocab_size   = VOCAB_SIZE,
    embed_dim    = 128,
    n_filters    = 100,
    filter_sizes = [3, 4, 5],
    output_dim   = NUM_CLASSES,
    dropout      = 0.5
).to(device)

total_params = sum(p.numel() for p in model_kim.parameters() if p.requires_grad)
print(f"  Tổng số tham số: {total_params:,}")

# ── Train ─────────────────────────────────────────────────────────
model_kim, history, test_preds, test_labels = train_seq_model(
    model        = model_kim,
    train_loader = trn_seq,
    val_loader   = val_seq,
    test_loader  = tst_seq,
    label_map    = label_map,
    model_name   = "S1 — KimCNN (Baseline)",
    device       = device,
    epochs       = 12,
    lr           = 1e-3,
    patience     = 3,
)

# ── Kết quả ───────────────────────────────────────────────────────
acc, mf1, wf1 = print_metrics(test_labels, test_preds, label_map, "S1 KimCNN")
plot_results(history, test_labels, test_preds, label_map, "S1: KimCNN", cmap='Blues')
from Processing.Loader_Data import test_df
run_robustness_seq(model_kim, test_df, vocab, label_map, device, model_name="KimCNN")
save_dir = os.path.join(root_path, "Saved_Models")
os.makedirs(save_dir, exist_ok=True)
torch.save(model_kim.state_dict(), os.path.join(save_dir, "kimcnn_s1.pt"))
print("✅ Saved: kimcnn_s1.pt")