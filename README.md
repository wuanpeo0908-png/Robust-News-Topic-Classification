# 🗞️ Robust News Topic Classification

> **A Multi-Dimensional Robustness Study on Vietnamese News Data**  
> Phenikaa University — Natural Language Processing, 2025–2026

---

## 📌 Overview

This project evaluates the **robustness** of five deep learning architectures for **Vietnamese news topic classification** across three experimental scenarios. Beyond standard accuracy, we systematically test each model against real-world challenges: noisy input, domain shift, and document length sensitivity.

**Author:** Minh Quan Nguyen Vu (22011082)  
**Advisor:** ThS. Vu Hoang Dieu  
**Course:** Natural Language Processing — Phenikaa University  
**Date:** March 2026

---

## 🧠 Models Evaluated

| Model | Type |
|---|---|
| **KimCNN** | Parallel CNN with max-over-time pooling |
| **BiLSTM + Attention** | Bidirectional LSTM with additive attention |
| **RCNN** | BiLSTM outputs + word embeddings + max pooling |
| **Transformer Encoder** | 2-layer encoder with learned positional encodings |
| **PhoBERT** | `vinai/phobert-base` — 12 layers, 110M parameters |

---

## 📂 Project Structure

```
Robust-News-Topic-Classification/
│
├── src/                        # Source code
│   ├── imports.py              # Shared imports and dependencies
│   ├── S1_Rubustness_Test.py   # Scenario 1 robustness evaluation (VnExpress)
│   └── S2_Rubustness_Test.py   # Scenario 2 robustness evaluation (NamSyntax)
│
├── data/
│   ├── raw/                    # S1 — VnExpress self-collected articles
│   └── processed/              # S2 — NamSyntax/HuggingFace preprocessed data
│
├── models/                     # Saved model weights (.pkl, .pt, etc.)
│
├── results/                    # Evaluation outputs, charts, tables
│
├── scripts/                    # Experiment scripts (Các Kịch Bản)
│
├── docs/
│   └── NLP_2.pdf               # Full research paper
│
├── .gitignore
└── README.md
```

---

## 📊 Datasets

### S1 — VnExpress (Self-collected)
- **5,712 articles** crawled from VnExpress.net
- **10 categories:** Politics, World, Business, Science, Real Estate, Health, Sports, Entertainment, Law, Education
- Split: 80% train / 10% val / 10% test
- Avg. length: 200–500 words

### S2 — NamSyntax / HuggingFace
- Sampled from `NamSyntax/vietnamese-news-classification` (1.3M articles)
- **5,957 samples** after preprocessing (600/class)
- Same 10-category mapping as S1

### S3 — Cross-Domain
- S2-trained weights tested on S1 data (zero-shot, no retraining)

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/wuanpeo0908-png/Robust-News-Topic-Classification.git
cd Robust-News-Topic-Classification

# Install dependencies
pip install torch transformers underthesea scikit-learn pandas numpy matplotlib
```

**Hardware used:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)  
**Software:** Python 3.11, PyTorch, HuggingFace Transformers, underthesea

---

## 🚀 Running the Experiments

```bash
# Run Scenario 1 robustness test (VnExpress)
python src/S1_Rubustness_Test.py

# Run Scenario 2 robustness test (NamSyntax)
python src/S2_Rubustness_Test.py
```

---

## 📈 Key Results

### In-Domain Performance

| Model | S1 Accuracy | S2 Accuracy |
|---|---|---|
| KimCNN | 87.4% | 63.6% |
| BiLSTM+Attention | 71.0% | 57.4% |
| RCNN | 89.7% | 63.8% |
| Transformer | 85.1% | 65.8% |
| **PhoBERT** | **94.6%** | **81.4%** |

### Noise Robustness (S1 — Accuracy)

| Model | Clean | Typo | No Accent | Drop Words |
|---|---|---|---|---|
| KimCNN | 87.6% | 87.6% | 37.2% | 87.6% |
| BiLSTM+Attention | 66.3% | 66.6% | 15.4% | 63.1% |
| RCNN | 91.6% | 90.4% | 18.2% | 89.9% |
| Transformer | 85.8% | 83.2% | 12.6% | 84.3% |
| **PhoBERT** | **94.9%** | **93.7%** | 24.8% | **93.4%** |

### Cross-Domain Generalization (S3)

| Model | S1 Acc | S3 Acc | Drop |
|---|---|---|---|
| KimCNN | 87.4% | 73.3% | −16.2% |
| BiLSTM+Attention | 71.0% | 59.8% | −15.8% |
| RCNN | 89.7% | 66.6% | −25.8% |
| Transformer | 85.1% | 46.0% | −46.0% |
| **PhoBERT** | **94.6%** | **~88.0%** | **−7.0%** |

---

## 🔍 Key Findings

- **PhoBERT** achieves the highest accuracy across all scenarios and best noise resilience overall
- **KimCNN** is the strongest non-pretrained model for cross-domain generalization (S3: 73.3%)
- **Diacritic removal is catastrophic** for all models — Transformer drops −85.3%, no model retains above 40% accuracy
- **Short texts (<100 words)** cause near-random accuracy (≈50%) for most models
- A **diacritic restoration preprocessing step** is mandatory for any real-world Vietnamese NLP deployment

---

## 🛡️ Robustness Evaluation Framework

Three noise types applied independently to each test set:

| Perturbation | Method |
|---|---|
| **Typo** | 10% probability of adjacent-character transposition per word |
| **No Accent** | Diacritical marks removed via Unicode NFD decomposition |
| **Drop Words** | Each token deleted with probability 0.3 |

Document length strata: **Short** (<100 words), **Medium** (100–300 words), **Long** (>300 words)

---

## 📚 References

1. Y. Kim, "Convolutional Neural Networks for Sentence Classification," *EMNLP*, 2014.
2. D. Q. Nguyen, "PhoBERT: Pre-trained Language Models for Vietnamese," *Findings of ACL*, 2020.
3. NamSyntax, "Vietnamese News Classification Dataset," HuggingFace, 2023.

---

## 📄 License

This project is for academic purposes at Phenikaa University. See `docs/NLP_2.pdf` for the full research paper.
