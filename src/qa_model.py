import re
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

VALID_CHOICES = {"A", "B", "C", "D"}
CHOICE_PREFIX_RE = re.compile(r"^[A-D]\.\s*")


def normalize_label(x):
    return unicodedata.normalize("NFC", str(x).strip())


def build_law_lookup(law_items):
    return {(it["law_id"], str(it["article_id"])): it["content"] for it in law_items}


def build_content_vectorizer(law_items, max_law_chars=1200, max_features=20000):
    """TF-IDF vectorizer fit trên corpus luật, dùng để tính cosine-similarity làm đặc trưng
    cho ChoiceScorer — độc lập với retriever dùng để xếp hạng (TF-IDF hoặc BM25)."""
    texts = [(it["content"] or "")[:max_law_chars] for it in law_items]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    vectorizer.fit(texts)
    return vectorizer


def clean_choice_text(text):
    return CHOICE_PREFIX_RE.sub("", str(text or "").strip())


def jaccard_overlap(a, b):
    ta = set(re.findall(r"\w+", (a or "").lower(), flags=re.UNICODE))
    tb = set(re.findall(r"\w+", (b or "").lower(), flags=re.UNICODE))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(vectorizer, text_a, text_b):
    va = vectorizer.transform([text_a or ""])
    vb = vectorizer.transform([text_b or ""])
    return float(cosine_similarity(va, vb)[0, 0])


def context_text(example, retriever, law_lookup, top_k=3, max_ctx_chars=1200):
    """Ngữ cảnh luật cho 1 câu hỏi: dùng relevant_articles có sẵn nếu có (vd public_task2),
    ngược lại tự chạy retriever."""
    gold = example.get("relevant_articles")
    if gold:
        hits = gold
    else:
        hits = retriever.search(example.get("question", ""), top_k=top_k)
    texts = [law_lookup.get((h.get("law_id"), str(h.get("article_id"))), "") for h in hits]
    return " ".join(t for t in texts if t)[:max_ctx_chars]


class YesNoClassifier:
    def __init__(self, C=1.0):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)

    def fit(self, questions, labels):
        y = [1 if normalize_label(l) == "Đúng" else 0 for l in labels]
        X = self.vectorizer.fit_transform(questions)
        self.clf.fit(X, y)
        return self

    def predict(self, questions):
        X = self.vectorizer.transform(questions)
        return ["Đúng" if p == 1 else "Sai" for p in self.clf.predict(X)]

    def save(self, path):
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, path)

    @staticmethod
    def load(path):
        obj = joblib.load(path)
        model = YesNoClassifier()
        model.vectorizer = obj["vectorizer"]
        model.clf = obj["clf"]
        return model


def mc_features(example, retriever, law_lookup, vectorizer, top_k=3, max_ctx_chars=1200):
    """Sinh 1 dict đặc trưng cho mỗi choice (A-D) của 1 câu hỏi multiple-choice."""
    q = example.get("question", "")
    ctx = context_text(example, retriever, law_lookup, top_k=top_k, max_ctx_chars=max_ctx_chars)
    hits = example.get("relevant_articles") or retriever.search(q, top_k=top_k)
    top1_score = float(hits[0].get("score", 0.0)) if hits else 0.0
    rows = []
    for letter in ["A", "B", "C", "D"]:
        choice_raw = (example.get("choices") or {}).get(letter, "")
        choice = clean_choice_text(choice_raw)
        feat = {
            "cos_choice_ctx": _cosine(vectorizer, choice, ctx),
            "cos_choice_q": _cosine(vectorizer, choice, q),
            "overlap_choice_ctx": jaccard_overlap(choice, ctx),
            "overlap_choice_q": jaccard_overlap(choice, q),
            "choice_len_chars": float(len(choice)),
            "choice_num_tokens": float(len(choice.split())),
            "top1_retrieval_score": top1_score,
        }
        rows.append((letter, feat))
    return rows


class ChoiceScorer:
    def __init__(self, C=1.0, use_letter_onehot=False):
        self.dict_vec = DictVectorizer(sparse=False)
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        self.use_letter_onehot = use_letter_onehot

    def _augment(self, letter, feat):
        if not self.use_letter_onehot:
            return feat
        feat = dict(feat)
        for l in VALID_CHOICES:
            feat[f"letter_{l}"] = 1.0 if letter == l else 0.0
        return feat

    def fit(self, rows):
        """rows: list of (letter, feat_dict, label_int)."""
        X_dicts = [self._augment(letter, feat) for letter, feat, _ in rows]
        y = [label for _, _, label in rows]
        X = self.dict_vec.fit_transform(X_dicts)
        self.clf.fit(X, y)
        return self

    def predict_example(self, example, retriever, law_lookup, vectorizer, top_k=3, max_ctx_chars=1200):
        feat_rows = mc_features(example, retriever, law_lookup, vectorizer, top_k=top_k, max_ctx_chars=max_ctx_chars)
        X_dicts = [self._augment(letter, feat) for letter, feat in feat_rows]
        X = self.dict_vec.transform(X_dicts)
        proba = self.clf.predict_proba(X)[:, list(self.clf.classes_).index(1)]
        best_idx = int(np.argmax(proba))
        return feat_rows[best_idx][0]

    def save(self, path):
        joblib.dump({
            "dict_vec": self.dict_vec,
            "clf": self.clf,
            "use_letter_onehot": self.use_letter_onehot,
        }, path)

    @staticmethod
    def load(path):
        obj = joblib.load(path)
        model = ChoiceScorer(use_letter_onehot=obj["use_letter_onehot"])
        model.dict_vec = obj["dict_vec"]
        model.clf = obj["clf"]
        return model
