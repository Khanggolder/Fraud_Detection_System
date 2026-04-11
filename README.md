# Fraud Detection System

Hệ thống phát hiện gian lận mã nguồn đa ngôn ngữ (Python, C, C++) bao gồm: kiểm tra đạo văn, đo tương đồng ngữ nghĩa, và nhận diện code do AI tạo ra.

Hỗ trợ 2 engine phát hiện: **Rule-Based** (signal weights) và **Machine Learning** (Random Forest / XGBoost).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py        # Python detector
streamlit run app_c.py      # C detector
streamlit run app_cpp.py    # C++ detector
```

## Hệ thống phát hiện

### 1. Python Fraud Detection (`app.py`)

Phân tích file Python trên 3 nhánh độc lập:

- **Plagiarism Detection**: Winnowing fingerprint + Jaccard similarity
- **Semantic Similarity**: CodeBERT embedding + cosine similarity
- **AI Code Detection**: 19 signal functions + Qwen2.5-Coder perplexity scoring

| Module | Chức năng |
|--------|-----------|
| `src/preprocessor.py` | Chuẩn hóa AST cho plagiarism |
| `src/detectors.py` | Winnowing fingerprint + Jaccard |
| `src/semantic.py` | CodeBERT embedding + cosine similarity |
| `src/features.py` | Trích xuất 70+ features Python |
| `src/ai_detector.py` | 19 signals + Qwen2.5-Coder perplexity |

---

### 2. C AI Code Detector (`app_c.py`)

Phát hiện code C do AI tạo ra bằng phân tích AST (Tree-sitter).

- **Features**: 106 features (memory management, header analysis, naming, formatting, Halstead)
- **Signals**: 10 signal functions + Token Distribution (Qwen2.5-Coder)
- **Dataset**: 112,001 AI + 1,099 Human files

| Module | Chức năng |
|--------|-----------|
| `src/c_features.py` | Trích xuất 106 features C |
| `src/c_ai_detector.py` | 10 signals + perplexity scoring |

---

### 3. C++ AI Code Detector (`app_cpp.py`)

Phát hiện code C++ do AI tạo ra bằng phân tích AST (Tree-sitter).

- **Features**: 134 features (modern C++, RAII, smart pointers, naming, formatting, Halstead)
- **Signals**: 11 signal functions + Token Distribution (Qwen2.5-Coder)
- **Dataset**: 154 AI + 617 Human files

| Module | Chức năng |
|--------|-----------|
| `src/cpp_features.py` | Trích xuất 134 features C++ |
| `src/cpp_ai_detector.py` | 11 signals + perplexity scoring |

---

### 4. ML Pipeline (Chung cho C và C++)

Thay thế rule-based bằng model ML đã train sẵn.

| Module | Chức năng |
|--------|-----------|
| `extract_features_csv.py` | Batch extraction (multiprocessing, checkpointing) |
| `train_model.py` | Train XGBoost / Random Forest với 5-Fold CV |
| `src/ml_detector.py` | Inference bằng model .pkl |

## Kết quả Test trên Dataset thực tế

Test được thực hiện bằng `test_all_systems.py` trên dataset thực, lấy mẫu ngẫu nhiên tối đa 200 file/class (seed=42).

### Tổng hợp kết quả

| Hệ thống | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|----------|----------|-----------|--------|----------|----|----|----|----|
| **Python Rule-Based** | 64.67% | 0.4615 | 0.3600 | 0.4045 | 18 | 21 | 79 | 32 |
| **C Rule-Based** | 94.50% | 0.9890 | 0.9000 | 0.9424 | 180 | 2 | 198 | 20 |
| **C++ Rule-Based** | 75.71% | 1.0000 | 0.4416 | 0.6126 | 68 | 0 | 200 | 86 |
| **C ML Model (RF)** | **100.00%** | 1.0000 | 1.0000 | 1.0000 | 200 | 0 | 200 | 0 |
| **C++ ML Model (RF)** | **99.72%** | 1.0000 | 0.9935 | 0.9967 | 153 | 0 | 200 | 1 |

### Chi tiết từng hệ thống

#### Python Rule-Based (50 AI + 100 Human, threshold=0.60)

- Accuracy: 64.67% — F1: 0.4045
- Precision: 0.46 (nhiều false positive)
- Recall: 0.36 (bỏ sót nhiều AI code)
- Dataset Python nhỏ (50 AI + 100 Human), chưa có ML model

#### C Rule-Based (200 AI + 200 Human, threshold=0.35)

- Accuracy: 94.50% — F1: 0.9424
- Precision: 0.989 (gần như không báo nhầm)
- Recall: 0.90 (phát hiện được 90% AI code)
- Tốc độ: 69 files/sec

#### C++ Rule-Based (154 AI + 200 Human, threshold=0.60)

- Accuracy: 75.71% — F1: 0.6126
- Precision: 1.000 (không có false positive)
- Recall: 0.4416 (bỏ sót 56% AI code — cần hạ threshold)
- Tốc độ: 65 files/sec

#### C ML Model — Random Forest (200 AI + 200 Human, threshold=0.50)

- Accuracy: 100.00% — F1: 1.0000
- TP=200, FP=0, TN=200, FN=0
- Tốc độ: 8 files/sec (chậm hơn do feature extraction)

#### C++ ML Model — Random Forest (154 AI + 200 Human, threshold=0.50)

- Accuracy: 99.72% — F1: 0.9967
- TP=153, FP=0, TN=200, FN=1 (chỉ miss 1 file)
- Tốc độ: 7 files/sec

### ML Model Training Metrics

#### C Model (Random Forest)

| Metric | 5-Fold CV | Full Dataset |
|--------|-----------|--------------|
| Accuracy | 1.0000 ± 0.0000 | 1.0000 |
| F1 Score | 1.0000 ± 0.0000 | 1.0000 |
| ROC-AUC | 1.0000 ± 0.0000 | 1.0000 |

- Training: 1,000 samples (500 AI + 500 Human) — 106 features

#### C++ Model (Random Forest)

| Metric | 5-Fold CV | Full Dataset |
|--------|-----------|--------------|
| Accuracy | 1.0000 ± 0.0000 | 1.0000 |
| F1 Score | 1.0000 ± 0.0000 | 1.0000 |
| ROC-AUC | 1.0000 ± 0.0000 | 1.0000 |

- Training: 100 samples (50 AI + 50 Human) — 134 features

### Overfitting Diagnostic

| Bài Test | C | C++ |
|----------|---|-----|
| Train/Test Gap (80/20) | 0.00% | 0.00% |
| Learning Curve | Hội tụ | Hội tụ |
| 10-Fold CV Mean Accuracy | 99.97% | 100.00% |
| Permutation Test (Shuffled) | 60.4% | 74.5% |

### Top Features

**C:**

| Feature | Importance |
|---------|------------|
| `std_header_count` | 0.857 |
| `op_spacing_rate` | 0.800 |
| `header_diversity` | 0.719 |
| `comma_space_rate` | 0.711 |
| `ast_error_ratio` | 0.676 |

**C++:**

| Feature | Importance |
|---------|------------|
| `specific_header_count` | 0.950 |
| `unique_headers` | 0.950 |
| `specific_header_ratio` | 0.920 |
| `modern_cpp_ratio` | 0.812 |
| `brace_same_line_count` | 0.788 |

## Cấu trúc thư mục

```
fraud_detection_system/
  app.py                     # UI Python detector
  app_c.py                   # UI C detector
  app_cpp.py                 # UI C++ detector
  extract_features_csv.py    # Batch feature extraction
  train_model.py             # ML training pipeline
  test_all_systems.py        # Test script cho toàn bộ hệ thống
  check_overfitting.py       # Overfitting diagnostic
  requirements.txt
  README.md
  data/                      # Dataset Python (50 AI + 100 Human)
  data_C/                    # Dataset C (112,001 AI + 1,099 Human)
  dataset_CPP/               # Dataset C++ (154 AI + 617 Human)
  models/                    # Trained ML models
    c_model.pkl / c_scaler.pkl / c_metadata.json
    cpp_model.pkl / cpp_scaler.pkl / cpp_metadata.json
  src/
    preprocessor.py          # AST normalization (Python)
    detectors.py             # Winnowing + Jaccard
    semantic.py              # CodeBERT + cosine similarity
    features.py              # Python features (70+)
    ai_detector.py           # Python signals (19) + Qwen2.5-Coder
    c_features.py            # C features (106)
    c_ai_detector.py         # C signals (10)
    cpp_features.py          # C++ features (134)
    cpp_ai_detector.py       # C++ signals (11)
    ml_detector.py           # ML inference module
```

## Detection Modes

| Mode | Engine | Ưu điểm | Nhược điểm |
|------|--------|---------|-------------|
| Rule-Based | Signal weights + sigmoid | Giải thích được, nhanh | Cần calibrate thủ công |
| ML Model | Random Forest / XGBoost | Accuracy cao nhất | Cần dataset + training |

## Perplexity Model

Cả 3 hệ thống sử dụng **Qwen/Qwen2.5-Coder-0.5B** (optional):
- AI code = perplexity thấp (dễ đoán)
- Human code = perplexity cao (khó đoán)
- Cần ~1GB download lần đầu

## Dependencies

```
streamlit, pandas, numpy
matplotlib, seaborn, networkx
transformers, torch, scikit-learn, xgboost, joblib
radon
```
