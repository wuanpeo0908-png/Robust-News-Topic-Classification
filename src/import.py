# =================================================================
# FILE: imports.py - TỔNG HỢP THƯ VIỆN & CẤU HÌNH GPU (Scenario 1)
# =================================================================

import os
import re
import json
import math
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# --- Deep Learning & NLP (PyTorch & Transformers) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Giải pháp cho PhoBERT và lỗi AdamW
from pyvi import ViTokenizer
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW 

# --- Machine Learning Metrics & Tools ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Thanh tiến độ ---
from tqdm import tqdm

# --- Cấu hình thiết bị GPU (Dành cho card RTX 3050 của DL) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Cấu hình hiển thị biểu đồ ---
plt.rcParams['figure.figsize'] = (15, 5)
sns.set(style='whitegrid')

print(f"✅ Thư viện đã nạp thành công!")
print(f"🚀 Đang chạy trên: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU hiện tại: {torch.cuda.get_device_name(0)}")
