import sys
import os
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- BƯỚC 1: KẾT NỐI HỆ THỐNG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from imports import device
from Processing.Structure_5models   import KimCNN, RCNN, TransformerClassifier, BiLSTM_Attention, PhoBertClassifier
from Processing.train_evaluate      import evaluate_seq, print_metrics
from Processing.Vocabulary_LabelMap import VOCAB_SIZE, label_map
from Processing.Loader_Data         import tst_seq as s1_test_loader, tst_pho

os.makedirs(os.path.join(root_path, "Results"), exist_ok=True)

# --- BƯỚC 2: ĐƯỜNG DẪN WEIGHTS S2 ---
s2_weights = {
    'KimCNN'      : 'Saved_Models/kimcnn_s2.pt',
    'RCNN'        : 'Saved_Models/rcnn_s2.pt',
    'Transformer' : 'Saved_Models/transformer_s2.pt',
    'BiLSTM-Attn' : 'Saved_Models/bilstm_attention_s2.pt',
    'PhoBERT'     : 'Saved_Models/phobert_s2.pt',
}

# --- HÀM EVAL PHOBERT ---
def eval_pho(model, loader, device):
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
    return labels, preds

# --- BƯỚC 3: HÀM CHẠY ---
def run_s3_robustness():
    results_s3   = []
    criterion    = torch.nn.CrossEntropyLoss()
    target_names = [str(k) for k in sorted(label_map.keys())]

    for name, path_suffix in s2_weights.items():
        weight_path = os.path.join(root_path, path_suffix)
        if not os.path.exists(weight_path):
            print(f"⚠️ Thiếu file: {weight_path}")
            continue

        print(f"\n🚀 Đang đánh giá: {name}")

        # Khởi tạo model đúng cấu hình S2
        if name == 'KimCNN':
            model = KimCNN(VOCAB_SIZE, 128, 100, [3,4,5], 10, 0.5).to(device)
        elif name == 'RCNN':
            model = RCNN(VOCAB_SIZE, 128, 256, 10, dropout=0.5).to(device)
        elif name == 'Transformer':
            model = TransformerClassifier(VOCAB_SIZE, 128, 8, 256, 2, 10, 0.3).to(device)
        elif name == 'BiLSTM-Attn':
            model = BiLSTM_Attention(VOCAB_SIZE, 128, 128, 10, n_layers=2, dropout=0.5).to(device)
        elif name == 'PhoBERT':
            model = PhoBertClassifier(output_dim=10, dropout=0.1).to(device)

        # Load weights S2
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        # Evaluate — PhoBERT dùng loader riêng
        if name == 'PhoBERT':
            y_true, y_pred = eval_pho(model, tst_pho, device)
        else:
            y_true, y_pred, _ = evaluate_seq(model, s1_test_loader, criterion, device)

        # Metrics
        acc, mf1, wf1 = print_metrics(y_true, y_pred, label_map, f"S3-{name}")
        results_s3.append([name, acc, mf1, wf1])

        # Confusion Matrix
        cm_mat = confusion_matrix(y_true, y_pred)
        cmn    = cm_mat.astype('float') / cm_mat.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title(f"S3 Robustness: {name} — Confusion Matrix (%)")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        save_path = os.path.join(root_path, f"Results/s3_cm_{name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: {save_path}")

    return pd.DataFrame(results_s3, columns=['Model', 'S3_Acc', 'S3_MacroF1', 'S3_WeightF1'])


# --- MAIN ---
if __name__ == "__main__":
    df_s3 = run_s3_robustness()
    print("\n" + "="*50)
    print("📊 BẢNG TỔNG HỢP ĐỘ ROBUST (TEST TRÊN S1)")
    print(df_s3.to_string(index=False))
    csv_path = os.path.join(root_path, "Results/s3_robust_results.csv")
    df_s3.to_csv(csv_path, index=False)
    print(f"\n✅ Đã lưu: {csv_path}")
