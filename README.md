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

## Kết quả Test

### Rule-Based Detector Test

| Hệ thống | Input | Score | Kết quả |
|-----------|-------|-------|---------|
| Python (`ai_detector.py`) | `def hello(): pass` | 36/100 | Human |
| C (`c_ai_detector.py`) | `int main() { return 0; }` | 22/100 | Human |
| C++ (`cpp_ai_detector.py`) | `#include <iostream> int main() {}` | 11/100 | Human |

### ML Model Test (Random Forest)

| Hệ thống | Input | Score | Kết quả |
|-----------|-------|-------|---------|
| ML C | `int main() { return 0; }` | 0/100 | Human |
| ML C++ | `#include <iostream> int main() {}` | 43/100 | Human |

### C Model — Training Metrics

| Metric | 5-Fold CV | Full Dataset |
|--------|-----------|--------------|
| Accuracy | 1.0000 ± 0.0000 | 1.0000 |
| Precision | — | 1.0000 |
| Recall | — | 1.0000 |
| F1 Score | 1.0000 ± 0.0000 | 1.0000 |
| ROC-AUC | 1.0000 ± 0.0000 | 1.0000 |

- Training samples: 1,000 (500 AI + 500 Human)
- Features: 106
- Model: Random Forest

### C++ Model — Training Metrics

| Metric | 5-Fold CV | Full Dataset |
|--------|-----------|--------------|
| Accuracy | 1.0000 ± 0.0000 | 1.0000 |
| Precision | — | 1.0000 |
| Recall | — | 1.0000 |
| F1 Score | 1.0000 ± 0.0000 | 1.0000 |
| ROC-AUC | 1.0000 ± 0.0000 | 1.0000 |

- Training samples: 100 (50 AI + 50 Human)
- Features: 134
- Model: Random Forest

### Overfitting Diagnostic

| Bài Test | C | C++ |
|----------|---|-----|
| Train/Test Gap (80/20) | 0.00% | 0.00% |
| Learning Curve | Hội tụ | Hội tụ |
| 10-Fold CV Mean Accuracy | 99.97% | 100.00% |
| 10-Fold CV Min Accuracy | 99.68% | 100.00% |
| Permutation Test (Shuffled) | 60.4% | 74.5% |

### Top Features

**C:**

| Feature | Correlation |
|---------|-------------|
| `std_header_count` | 0.857 |
| `op_spacing_rate` | 0.800 |
| `header_diversity` | 0.719 |
| `comma_space_rate` | 0.711 |
| `ast_error_ratio` | 0.676 |

**C++:**

| Feature | Correlation |
|---------|-------------|
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
  requirements.txt
  README.md
  data/                      # Dataset Python
  data_C/                    # Dataset C (112,001 AI + 1,099 Human)
    AI/
    Human/
  dataset_CPP/               # Dataset C++ (154 AI + 617 Human)
    ai/
    human/
  models/                    # Trained ML models
    c_model.pkl
    c_scaler.pkl
    c_metadata.json
    cpp_model.pkl
    cpp_scaler.pkl
    cpp_metadata.json
  src/
    preprocessor.py          # AST normalization (Python)
    detectors.py             # Winnowing + Jaccard
    semantic.py              # CodeBERT + cosine similarity
    features.py              # Python feature extraction (70+)
    ai_detector.py           # Python AI signals (19) + Qwen2.5-Coder
    c_features.py            # C feature extraction (106)
    c_ai_detector.py         # C AI signals (10)
    cpp_features.py          # C++ feature extraction (134)
    cpp_ai_detector.py       # C++ AI signals (11)
    ml_detector.py           # ML inference module
```

## Detection Modes

| Mode | Engine | Ưu điểm | Nhược điểm |
|------|--------|---------|-------------|
| Rule-Based | Signal weights + sigmoid | Giải thích được, không cần train | Cần calibrate thủ công |
| ML Model | Random Forest / XGBoost | Accuracy cao, tự học từ data | Cần dataset đủ lớn |

## Perplexity Model

Cả 3 hệ thống đều sử dụng **Qwen/Qwen2.5-Coder-0.5B** cho perplexity scoring (optional):
- AI code có perplexity thấp (dễ đoán)
- Human code có perplexity cao (khó đoán hơn)
- Burstiness thấp = output đều đặn = AI pattern
- Cần ~1GB download lần đầu

## Dependencies

```
streamlit, pandas, numpy
matplotlib, seaborn, networkx
transformers, torch, scikit-learn, xgboost, joblib
radon
```
