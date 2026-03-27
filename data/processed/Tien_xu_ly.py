# %% Cell 3: Tiền xử lý & Tách từ (Tokenization)
import pandas as pd
import re
from underthesea import word_tokenize # Đã sửa: bỏ 'cls'
from tqdm import tqdm
import os

# Đường dẫn file
file_path = r'D:\Phenikaa-Study\kì 2 2025-2026\Xử lý ngôn ngữ\VS code\Thu Data\data.csv'
output_path = r'D:\Phenikaa-Study\kì 2 2025-2026\Xử lý ngôn ngữ\VS code\Processing\data_ready_for_model.csv'

# Tạo thư mục Processing nếu chưa có để tránh lỗi lưu file
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(file_path)

def clean_text_robust(text):
    # Sửa lỗi: Nếu không phải string thì trả về chuỗi rỗng thay vì None
    if not isinstance(text, str): 
        return ""
    
    # 1. Chuẩn hóa chữ thường và xóa ký tự đặc biệt/số
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # 2. Tách từ tiếng Việt (Word Segmentation)
    # format="text" biến "kinh doanh" -> "kinh_doanh"
    text = word_tokenize(text, format="text")
    
    # 3. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("--- ĐANG BẮT ĐẦU TIỀN XỬ LÝ VĂN BẢN (Tokenization) ---")
# Sử dụng tqdm để theo dõi tiến độ
tqdm.pandas()
df['text_clean'] = df['text'].progress_apply(clean_text_robust)

# 4. Lọc bỏ các bài báo quá ngắn (dưới 50 từ) hoặc bị rỗng
df = df[df['text_clean'].str.split().str.len() > 50]

# 5. Lưu file sạch cuối cùng
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\nHoàn thành! Số lượng mẫu sạch sau khi lọc: {len(df)}")
print(f"File đã sẵn sàng tại: {output_path}")