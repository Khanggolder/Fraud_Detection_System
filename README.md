# Fraud Detection System

Hệ thống phát hiện gian lận mã nguồn Python, bao gồm: kiểm tra đạo văn (plagiarism), đo tương đồng ngữ nghĩa (semantic similarity), và nhận diện code do AI tạo ra.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, mở trình duyệt và upload các file `.py` từ sidebar. Hệ thống sẽ phân tích trên 3 nhánh độc lập.

## Các nhánh phân tích

### 1. Kiểm tra đạo văn (Plagiarism Detection)

Sử dụng thuật toán Winnowing để tạo fingerprint từ mã nguồn đã được chuẩn hóa AST, rồi tính độ tương đồng Jaccard giữa các cặp file.

- Module: `src/detectors.py`
- Đầu vào: code đã normalize (qua `src/preprocessor.py`)
- Đầu ra: điểm tương đồng 0 - 1

### 2. Đo tương đồng ngữ nghĩa (Semantic Similarity)

Dùng mô hình CodeBERT để tạo embedding cho từng file, sau đó tính cosine similarity giữa các cặp.

- Module: `src/semantic.py`
- Đầu vào: code gốc
- Đầu ra: điểm tương đồng 0 - 1

### 3. Nhận diện code AI (AI Code Detection)

Phân tích phong cách lập trình (stylometry) để ước tính xác suất code được tạo bởi AI. Module này làm việc trên **code gốc** (raw code), không qua bất kỳ bước tiền xử lý nào để giữ nguyên các dấu hiệu về whitespace, comment, naming.

- Module: `src/features.py` (trích xuất đặc trưng) + `src/ai_detector.py` (tính điểm)
- Đầu vào: code gốc (raw string)
- Đầu ra:
  - `p_ai`: xác suất ước tính (0.0 - 1.0)
  - `score`: điểm số (0 - 100)
  - `flag`: True/False theo ngưỡng
  - `signals`: danh sách 3-5 tín hiệu chính

#### Các nhóm đặc trưng (features)

1. **Whitespace / Layout**: độ nhất quán của thụt lề, khoảng trắng toán tử, dấu phẩy, độ dài dòng, trailing whitespace.
2. **Comments**: tỷ lệ comment, docstring (đếm bằng AST), độ dài comment, tutorial markers (Args, Returns, Example...).
3. **Token / Style**: pythonic constructs, type hints, naming convention, số lượng hàm/class, error handling.
4. **Radon metrics**: cyclomatic complexity, maintainability index, Halstead volume/difficulty.

#### Cách tính điểm

Mỗi nhóm đặc trưng đóng góp các "tín hiệu" (signals) với trọng số (weight) khác nhau. Tổng trọng số được đưa qua hàm sigmoid để ra `p_ai`. Nếu `p_ai >= threshold` thì file bị đánh dấu (flag).

#### Ngưỡng mặc định

Threshold mặc định là **0.60**. Có thể chỉnh bằng slider trên giao diện.

- Tăng ngưỡng: giảm false positive (ít báo nhầm), nhưng có thể bỏ sót.
- Giảm ngưỡng: bắt nhiều hơn nhưng dễ báo nhầm.

## Cấu trúc thư mục

```
fraud_detection_system/
  app.py                  # Giao diện Streamlit
  requirements.txt
  README.md
  data/                   # File mẫu để test
    ai_generated.py
    original.py
    plagiarized.py
  src/
    preprocessor.py       # Chuẩn hóa AST (dùng cho plagiarism)
    detectors.py          # Winnowing fingerprint
    semantic.py           # CodeBERT embedding
    features.py           # Trích xuất 60+ đặc trưng stylometry
    ai_detector.py        # Tính điểm AI detection
```

## Ngưỡng cảnh báo (tham khảo)

| Chỉ số | Mức cảnh báo |
|--------|-------------|
| MOSS Similarity | > 0.7 là cao |
| Semantic Similarity | > 0.8 là cao |
| AI Score | Tùy theo threshold, mặc định >= 60 |
