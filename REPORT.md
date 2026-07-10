# Báo cáo: Nâng cấp MLQA-TSR Baseline bằng Học máy

Đồ án môn **Máy học nâng cao** — dựa trên bộ dữ liệu VLSP 2025 "MLQA-TSR" (Legal Question Answering
về luật giao thông và biển báo Việt Nam).

## 1. Giới thiệu & phạm vi

Bài toán gốc do BTC VLSP 2025 cung cấp gồm 2 track:
- **Task 1 — Retrieval**: cho một câu hỏi, truy hồi (các) điều luật liên quan trong kho luật giao thông.
- **Task 2 — Question Answering**: trả lời câu hỏi dạng Yes/No ("Đúng"/"Sai") hoặc trắc nghiệm 4 đáp án
  (A/B/C/D), dựa trên ngữ cảnh điều luật liên quan.

**Phạm vi đồ án**: chỉ xử lý nhánh văn bản (không xây dựng nhánh thị giác/CNN cho ảnh biển báo, dù dữ liệu
có kèm `image_id`). Mục tiêu là thay thế toàn bộ phần rule-based/hash của baseline gốc bằng các mô hình học
máy thật, có huấn luyện và đánh giá định lượng.

## 2. Dữ liệu

Nguồn: bộ dữ liệu VLSP 2025 do BTC cấp, gồm:
- `law_db/vlsp2025_law_new.json`: 2 văn bản luật, tổng cộng **402 điều luật** sau khi làm phẳng
  (`flatten_laws`).
- `train_data/vlsp_2025_train.json`: **530 câu hỏi có nhãn** — duy nhất tập có nhãn, dùng để train/val.
  Phân bố: 376 Multiple choice, 154 Yes/No.
- `public_test/*.json`: 50 câu hỏi cho Task 1 (chỉ có `question`, không nhãn) và 50 câu hỏi cho Task 2
  (có sẵn `relevant_articles` làm ngữ cảnh input, không có `answer`) — dùng để sinh submission cuối, không
  dùng để đánh giá vì không có nhãn.

### Vấn đề chất lượng dữ liệu đã xử lý
- 2 dạng Unicode khác nhau của nhãn "Đúng" (dạng tổ hợp dấu vs dựng sẵn) → chuẩn hoá bằng
  `unicodedata.normalize("NFC", ...)` ở mọi nơi so sánh nhãn (`qa_model.normalize_label`).
- 1 ví dụ (`train_112`) có `answer=40` (số, không hợp lệ cho multiple-choice) → loại khỏi tập huấn luyện,
  có log rõ khi chạy `scripts/train.py`.
- Một vài `choices` trong dữ liệu có kiểu `int` thay vì `str` → được ép kiểu an toàn trong
  `qa_model.clean_choice_text`.

### Bug nền tảng đã sửa trong data loader
Bản gốc `flatten_laws()` không parse được schema luật thật (`[{"id","title","articles":[...]}]`), nên
corpus luật rỗng và retriever chạy vô nghĩa trên dữ liệu thật của BTC. Đã viết lại để đệ quy vào `articles`
và dò đúng tên file (`vlsp2025_law_new.json`).

### Chia train/val
Vì `public_test` không có nhãn, toàn bộ việc huấn luyện + đánh giá dùng **530 ví dụ của `train_data`**,
chia stratify theo `question_type`: **423 train / 106 val** (`test_size=0.2, random_state=42`). Với
multiple-choice, việc sinh 4 dòng đặc trưng/câu hỏi được thực hiện **sau khi** chia theo id câu hỏi, để
tránh rò rỉ giữa các đáp án của cùng 1 câu hỏi giữa train và val.

## 3. Phương pháp

### 3.1 Retrieval (Task 1)
Giữ 2 phương pháp IR cổ điển để so sánh, cùng interface `.search(question, top_k)`:
- **TF-IDF + cosine similarity** (`src/retriever_tfidf.py`, đã có trong baseline gốc).
- **BM25** (`src/retriever_bm25.py`, dùng `rank_bm25`).

Retriever chỉ fit trên corpus luật (không phụ thuộc câu hỏi/nhãn) nên không có rủi ro rò rỉ; vector hoá
này được tái sử dụng an toàn để tính đặc trưng cho classifier ở mục 3.3.

### 3.2 Yes/No (Task 2a)
`YesNoClassifier` (`src/qa_model.py`): `TfidfVectorizer` (unigram+bigram, fit trên câu hỏi của **train
split**) + `LogisticRegression(class_weight="balanced")`. Thay thế hoàn toàn logic so khớp từ khoá cứng
của baseline gốc.

### 3.3 Multiple choice (Task 2b)
Vì đây là bài chọn 1/4 đáp án dạng văn bản tự do (không phải lớp cố định), dùng **pointwise scoring**:
với mỗi câu hỏi, sinh 4 dòng đặc trưng (1 dòng/đáp án) — cosine similarity giữa đáp án và ngữ cảnh luật,
cosine similarity giữa đáp án và câu hỏi, độ dài đáp án, keyword overlap (Jaccard), điểm retrieval top-1.
Huấn luyện `LogisticRegression` để chấm điểm "đúng/sai" từng dòng, dự đoán = đáp án có xác suất cao nhất
(`ChoiceScorer`). Thay thế hoàn toàn baseline chọn đáp án bằng `MD5(question) mod 4`.

**Ablation**: huấn luyện thêm 1 phiên bản có thêm đặc trưng one-hot vị trí đáp án (A/B/C/D), vì phân bố
nhãn thật lệch (A=114 > B=103 > C=84 > D=74).

## 4. Thực nghiệm & kết quả (trên tập val, 106 ví dụ, seed=42)

Chạy bằng `python scripts/evaluate.py --dataset-root "<path>" --models-dir models`.

### Bảng 1 — Retrieval (Recall@k, MRR)

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.170 | 0.274 | 0.311 | 0.226 |
| BM25 | 0.113 | 0.198 | 0.226 | 0.156 |

→ TF-IDF vượt BM25 trên corpus này (có thể do corpus luật nhỏ, ~400 điều, và câu hỏi ngắn — BM25 thường
lợi thế hơn trên corpus lớn/dài). Recall còn thấp (baseline IR đơn giản, không dùng embedding ngữ nghĩa)
nhưng **quan trọng nhất là khác 0**, xác nhận bug corpus rỗng đã được sửa đúng.

### Bảng 2 — Yes/No (31 ví dụ val)

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (keyword rules) | 0.613 | 0.380 |
| **YesNoClassifier (LogReg)** | **0.774** | **0.774** |

→ Cải thiện rõ rệt, đặc biệt Macro-F1 (0.38 → 0.77) cho thấy baseline cũ thiên vị nặng về 1 nhãn còn model
mới cân bằng giữa 2 lớp.

### Bảng 3 — Multiple choice (75 ví dụ val)

| Method | Accuracy |
|---|---|
| Baseline (MD5 hash) | 0.160 |
| **ChoiceScorer (không one-hot vị trí)** | **0.400** |
| ChoiceScorer (+one-hot vị trí, ablation) | 0.267 |

→ Model mới vượt xa baseline ngẫu nhiên (hash ≈ 0.16, thấp hơn cả 0.25 lý thuyết do mẫu nhỏ). Điểm thú
vị: thêm đặc trưng vị trí đáp án làm **giảm** accuracy trên val (0.40 → 0.267) — dấu hiệu overfit vào
"prior" vị trí đặc thù của tập train nhỏ (~300 ví dụ), không tổng quát hoá tốt. Đây là minh hoạ thực tế
cho hiện tượng overfitting khi thêm đặc trưng không có ý nghĩa nhân quả thật sự.

## 5. Hạn chế

- Chỉ có 530 ví dụ có nhãn — tập val (106 ví dụ, ~31 Yes/No, ~75 MC) khá nhỏ nên các con số accuracy có
  phương sai cao giữa các lần chia khác nhau.
- Không thể đánh giá trên `public_test` vì không có nhãn công khai.
- Multiple-choice mới dừng ở pointwise scoring, chưa phải learning-to-rank đầy đủ (chưa mô hình hoá tương
  quan giữa các đáp án trong cùng 1 câu hỏi).
- Chưa dùng sentence-embedding hay mô hình ngôn ngữ lớn — toàn bộ đặc trưng dựa trên TF-IDF cổ điển.

## 6. Hướng phát triển

- Retriever: dùng sentence-embedding đa ngôn ngữ (SBERT/PhoBERT) + FAISS, hoặc cross-encoder rerank sau
  bước retrieval thô.
- Đánh giá: k-fold cross-validation thay vì 1 lần chia train/val để giảm phương sai với tập nhỏ.
- Multiple-choice: chuyển sang learning-to-rank thật (listwise loss) thay vì pointwise.
- Nếu mở rộng phạm vi: kết hợp nhánh thị giác (CNN/ViT nhận diện biển báo từ `image_id`) cho bài toán
  multimodal QA đầy đủ đúng tinh thần "TSR" trong tên cuộc thi.

## 7. Cách chạy lại

```bash
pip install -r requirements.txt

python scripts/train.py --dataset-root "<path đến VLSP 2025 - MLQA-TSR Data Release>" --output-dir models
python scripts/evaluate.py --dataset-root "<path>" --models-dir models

# Sinh submission cho từng track
python scripts/predict.py --dataset-root "<path>" --split public_task1 --output-dir outputs_t1
python scripts/predict.py --dataset-root "<path>" --split public_task2 --output-dir outputs_t2
```

Xem `notebook/eda_train_demo.ipynb` để có EDA trực quan, quá trình huấn luyện từng bước và biểu đồ so
sánh baseline vs model mới.
