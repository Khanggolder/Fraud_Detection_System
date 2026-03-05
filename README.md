# Fraud Detection System

He thong phat hien gian lan ma nguon Python, bao gom: kiem tra dao van (plagiarism), do tuong dong ngu nghia (semantic similarity), va nhan dien code do AI tao ra.

## Cai dat

```bash
pip install -r requirements.txt
```

## Chay ung dung

```bash
streamlit run app.py
```

Sau khi chay, mo trinh duyet va upload cac file `.py` tu sidebar. He thong se phan tich tren 3 nhanh doc lap.

## Cac nhanh phan tich

### 1. Kiem tra dao van (Plagiarism Detection)

Su dung thuat toan Winnowing de tao fingerprint tu ma nguon da duoc chuan hoa AST, roi tinh do tuong dong Jaccard giua cac cap file.

- Module: `src/detectors.py`
- Dau vao: code da normalize (qua `src/preprocessor.py`)
- Dau ra: diem tuong dong 0 - 1

### 2. Do tuong dong ngu nghia (Semantic Similarity)

Dung mo hinh CodeBERT de tao embedding cho tung file, sau do tinh cosine similarity giua cac cap.

- Module: `src/semantic.py`
- Dau vao: code goc
- Dau ra: diem tuong dong 0 - 1

### 3. Nhan dien code AI (AI Code Detection)

Phan tich phong cach lap trinh (stylometry) de uoc tinh xac suat code duoc tao boi AI. Module nay lam viec tren **code goc** (raw code), khong qua bat ky buoc tien xu ly nao de giu nguyen cac dau hieu ve whitespace, comment, naming.

- Module: `src/features.py` (trich xuat dac trung) + `src/ai_detector.py` (tinh diem)
- Dau vao: code goc (raw string)
- Dau ra:
  - `p_ai`: xac suat uoc tinh (0.0 - 1.0)
  - `score`: diem so (0 - 100)
  - `flag`: True/False theo nguong
  - `signals`: danh sach 3-5 tin hieu chinh

#### Cac nhom dac trung (features)

1. **Whitespace / Layout**: do nhat quan cua thut le, khoang trang toan tu, dau phay, do dai dong, trailing whitespace.
2. **Comments**: ty le comment, docstring (dem bang AST), do dai comment, tutorial markers (Args, Returns, Example...).
3. **Token / Style**: pythonic constructs, type hints, naming convention, so luong ham/class, error handling.
4. **Radon metrics**: cyclomatic complexity, maintainability index, Halstead volume/difficulty.

#### Cach tinh diem

Moi nhom dac trung dong gop cac "tin hieu" (signals) voi trong so (weight) khac nhau. Tong trong so duoc dua qua ham sigmoid de ra `p_ai`. Neu `p_ai >= threshold` thi file bi danh dau (flag).

#### Nguong mac dinh

Threshold mac dinh la **0.60**. Co the chinh bang slider tren giao dien.

- Tang nguong: giam false positive (it bao nham), nhung co the bo sot.
- Giam nguong: bat nhieu hon nhung de bao nham.

## Cau truc thu muc

```
fraud_detection_system/
  app.py                  # Giao dien Streamlit
  requirements.txt
  README.md
  data/                   # File mau de test
    ai_generated.py
    original.py
    plagiarized.py
  src/
    preprocessor.py       # Chuan hoa AST (dung cho plagiarism)
    detectors.py          # Winnowing fingerprint
    semantic.py           # CodeBERT embedding
    features.py           # Trich xuat 60+ dac trung stylometry
    ai_detector.py        # Tinh diem AI detection
```

## Nguong canh bao (tham khao)

| Chi so | Muc canh bao |
|--------|-------------|
| MOSS Similarity | > 0.7 la cao |
| Semantic Similarity | > 0.8 la cao |
| AI Score | Tuy theo threshold, mac dinh >= 60 |
