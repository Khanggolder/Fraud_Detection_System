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
  - `signals`: danh sách tín hiệu chính (top 5)

#### Các nhóm đặc trưng (70+ features)

1. **Whitespace / Layout**: thụt lề, khoảng trắng toán tử, dấu phẩy, độ dài dòng, trailing whitespace, blank lines liên tiếp.
2. **Comments**: tỷ lệ comment, docstring (đếm bằng AST), perfect comment ratio, độ dài và độ lệch chuẩn comment, tutorial markers (Args, Returns, Example...).
3. **Token / Style**: pythonic constructs (enumerate, zip, comprehension, walrus...), type hints, naming convention, số lượng hàm/class, error handling.
4. **Human artifacts**: imbalanced spacing (x =3), range(len(...)), so sánh thừa (== True/False/None), dead code (code bị comment lại).
5. **Radon metrics**: cyclomatic complexity, maintainability index, Halstead volume/difficulty.

#### Hệ thống tín hiệu (19 signals)

Mỗi tín hiệu có trọng số (weight) khác nhau. Một số signal chính:

| Signal | Weight | Mô tả |
|--------|--------|-------|
| Clean Code Detection | 4.0 | Phát hiện code không có "lỗi người", sạch tuyệt đối |
| Human Artifacts Penalty | 3.5 | Trừ điểm khi có dấu hiệu code người (spacing lệch, range(len()), dead code) |
| Comment Style Analysis | 3.0 | Phân tích comment đều đặn, viết hoa chuẩn, std thấp |
| Tutorial Markers | 2.5 | Phát hiện Args/Returns/Example/Step trong docstring |
| Over-perfection Detection | 2.5 | Phát hiện code "hoàn hảo quá mức" đồng thời ở nhiều khía cạnh |
| Docstrings | 1.8 | Tỷ lệ hàm có docstring (đếm bằng AST) |
| Indent / Operator / Naming... | 0.8 - 1.5 | Các tín hiệu bổ trợ |

#### Cách tính điểm

Tổng weighted sum -> chuẩn hóa theo _MAX_WEIGHT -> scale [-3, +3] -> sigmoid -> `p_ai`.
Signal có thể trả giá trị âm (ví dụ: Human Artifacts Penalty), giúp giảm p_ai khi phát hiện code do người viết.

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
  data/
    ai_generated.py       # File mẫu AI-generated
    original.py           # File mẫu sinh viên viết
    plagiarized.py        # File mẫu đạo văn
    ai/                   # Thư mục chứa thêm mẫu AI
    human_pre_2021/       # Thư mục chứa mẫu code người (trước 2021)
  src/
    preprocessor.py       # Chuẩn hóa AST (dùng cho plagiarism)
    detectors.py          # Winnowing fingerprint + Jaccard similarity
    semantic.py           # CodeBERT embedding + cosine similarity
    features.py           # Trích xuất 70+ đặc trưng stylometry
    ai_detector.py        # 19 signals + sigmoid scoring + perplexity
```

## Ngưỡng cảnh báo (tham khảo)

| Chỉ số | Mức cảnh báo |
|--------|-------------|
| MOSS Similarity | > 0.7 là cao |
| Semantic Similarity | > 0.8 là cao |
| AI Score | Tùy theo threshold, mặc định >= 60 |
