"""
S2_2_Preprocess.py — Bước 2: Tiền xử lý văn bản
Đọc s2_raw.csv → tokenize bằng underthesea (giống S1) → lưu s2_clean.csv
Chạy SAU S2_1_LoadData.py, TRƯỚC S2_3_Evaluate.py
"""
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path   = os.path.dirname(os.path.dirname(current_dir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import re
import pandas as pd
from tqdm import tqdm
from underthesea import word_tokenize

DATA_DIR   = os.path.join(root_path, "S2_Data")
RAW_PATH   = os.path.join(DATA_DIR, "s2_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "s2_clean.csv")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy {RAW_PATH}\n→ Chạy S2_1_LoadData.py trước!")

# ════════════════════════════════════════════════════════════════
# HÀM TIỀN XỬ LÝ — giống hệt S1 (Tien_xu_ly.py)
# ════════════════════════════════════════════════════════════════
def clean_text_robust(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Lowercase + xóa ký tự đặc biệt
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # 2. Tách từ tiếng Việt (underthesea — giống S1)
    text = word_tokenize(text, format="text")
    # 3. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Đọc data ─────────────────────────────────────────────────────
print(f"⏳ Đọc: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"✅ Đã đọc: {len(df)} mẫu")

# ── Tiền xử lý ───────────────────────────────────────────────────
print("\n⏳ Tiền xử lý văn bản (underthesea — giống S1)...")
tqdm.pandas()
df['text_clean'] = df['text'].progress_apply(clean_text_robust)

# ── Lọc bài quá ngắn sau xử lý < 20 từ ──────────────────────────
before = len(df)
df = df[df['text_clean'].str.split().str.len() >= 20].reset_index(drop=True)
print(f"✅ Lọc bài ngắn: bỏ {before - len(df)}, còn {len(df)} mẫu")

# ── Thống kê ─────────────────────────────────────────────────────
wc = df['text_clean'].str.split().str.len()
print(f"\nĐộ dài sau xử lý (từ): mean={wc.mean():.0f} | min={wc.min()} | max={wc.max()}")
print("\nPhân bổ nhãn cuối:")
print(df['label'].value_counts().sort_index())

# ── Lưu ─────────────────────────────────────────────────────────
df[['text_clean', 'label']].to_csv(CLEAN_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu: {CLEAN_PATH}")
print("→ Chạy tiếp: S2_3_Evaluate.py")