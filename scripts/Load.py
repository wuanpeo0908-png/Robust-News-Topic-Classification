"""
S2_1_LoadData.py — Bước 1: Thu data & Mapping nhãn
Dataset: NamSyntax/vietnamese-news-classification (1.3M bài)

Nhãn HF vs S1 — khớp 1:1 (chỉ bỏ nhãn 10 Đời sống):
  0=Thời sự, 1=Thế giới, 2=Kinh doanh, 3=Khoa học CN,
  4=BĐS, 5=Sức khỏe, 6=Thể thao, 7=Giải trí,
  8=Pháp luật, 9=Giáo dục

Dùng cột 'text' (full content = title + description + body).
"""
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(os.path.dirname(current_dir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
from datasets import load_dataset

DATA_DIR      = os.path.join(root_path, "S2_Data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_SAVE_PATH = os.path.join(DATA_DIR, "s2_raw.csv")

LABEL_NAMES = {
    0: "Thời sự",      1: "Thế giới",      2: "Kinh doanh",
    3: "Khoa học CN",  4: "Bất động sản",  5: "Sức khỏe",
    6: "Thể thao",     7: "Giải trí",      8: "Pháp luật",
    9: "Giáo dục",
}

# ── Load ─────────────────────────────────────────────────────────
print("⏳ Load dataset từ HuggingFace...")
raw_ds = load_dataset("NamSyntax/vietnamese-news-classification")
df_all = pd.DataFrame(raw_ds['train'])

print(f"✅ Tổng mẫu gốc : {len(df_all):,}")
print(f"   Các cột      : {list(df_all.columns)}")
print(f"   Nhãn gốc     : {sorted(df_all['label'].unique())}")

# ── Bỏ nhãn 10 (Đời sống) ───────────────────────────────────────
df_all = df_all[df_all['label'] < 10].copy()
df_all['label'] = df_all['label'].astype(int)
print(f"\n✅ Sau khi bỏ nhãn 10: {len(df_all):,} mẫu")

# ── Dùng cột 'text' (full content) ──────────────────────────────
df_all = df_all[['text', 'label']].copy()
df_all['text'] = df_all['text'].fillna('').astype(str)

# ── Lọc bài quá ngắn < 30 từ ────────────────────────────────────
before = len(df_all)
df_all = df_all[df_all['text'].str.split().str.len() >= 30]
print(f"   Lọc bài <30 từ: bỏ {before - len(df_all):,}, còn {len(df_all):,}")

# ── Sampling 600 mẫu/nhãn → 6000 tổng ──────────────────────────
print("\n⏳ Sampling 600 mẫu/nhãn...")
df_sampled = df_all.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 600), random_state=42)
).reset_index(drop=True)

print(f"✅ Sampling xong: {len(df_sampled)} mẫu")
print("\nPhân bổ nhãn:")
for lbl, cnt in df_sampled['label'].value_counts().sort_index().items():
    print(f"   {lbl} ({LABEL_NAMES[lbl]:15s}): {cnt} mẫu")

# Thống kê độ dài nhanh
wc = df_sampled['text'].str.split().str.len()
print(f"\nĐộ dài văn bản (từ): mean={wc.mean():.0f} | min={wc.min()} | max={wc.max()}")

# ── Lưu ─────────────────────────────────────────────────────────
df_sampled.to_csv(RAW_SAVE_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu: {RAW_SAVE_PATH}")
print("→ Chạy tiếp: S2_2_Preprocess.py")