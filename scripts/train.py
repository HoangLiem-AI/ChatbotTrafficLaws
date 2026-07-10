import argparse
import json
import os
import sys

import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_from_root, load_from_zip, flatten_laws
from src.retriever_tfidf import TFIDFRetriever
from src.qa_model import (
    normalize_label,
    VALID_CHOICES,
    build_law_lookup,
    build_content_vectorizer,
    YesNoClassifier,
    ChoiceScorer,
    mc_features,
)


def build_retriever(law_items, method, top_k, max_law_chars):
    if method == "bm25":
        from src.retriever_bm25 import BM25Retriever
        return BM25Retriever(law_items, max_law_chars=max_law_chars)
    return TFIDFRetriever(law_items, max_law_chars=max_law_chars)


def load_corpus(dataset_root=None, dataset_zip=None):
    if dataset_root:
        laws, examples = load_from_root(dataset_root, split="train")
    elif dataset_zip:
        laws, examples = load_from_zip(dataset_zip, split="train")
    else:
        raise SystemExit("Cần --dataset-root hoặc --dataset-zip")
    law_items = flatten_laws(laws)
    return law_items, examples


def clean_train_examples(examples):
    """Loại các ví dụ multiple-choice có answer không thuộc {A,B,C,D}. Trả về (kept, dropped_ids)."""
    kept, dropped = [], []
    for ex in examples:
        qtype = (ex.get("question_type") or "").strip().lower()
        if qtype.startswith("multiple"):
            ans = normalize_label(ex.get("answer", "")).upper()
            if ans not in VALID_CHOICES:
                dropped.append(ex.get("id"))
                continue
        kept.append(ex)
    return kept, dropped


def split_examples(examples, val_size=0.2, seed=42):
    stratify = [ex.get("question_type") for ex in examples]
    train_ex, val_ex = train_test_split(examples, test_size=val_size, random_state=seed, stratify=stratify)
    return train_ex, val_ex


def train_yesno(examples):
    yn = [ex for ex in examples if (ex.get("question_type") or "").strip().lower() == "yes/no"]
    questions = [ex.get("question", "") for ex in yn]
    labels = [ex.get("answer", "") for ex in yn]
    model = YesNoClassifier().fit(questions, labels)
    return model, len(yn)


def train_mc(examples, retriever, law_lookup, vectorizer, use_letter_onehot, top_k, max_ctx_chars):
    mc = [ex for ex in examples if (ex.get("question_type") or "").strip().lower().startswith("multiple")]
    rows = []
    for ex in mc:
        gold_letter = normalize_label(ex.get("answer", "")).upper()
        for letter, feat in mc_features(ex, retriever, law_lookup, vectorizer, top_k=top_k, max_ctx_chars=max_ctx_chars):
            rows.append((letter, feat, 1 if letter == gold_letter else 0))
    model = ChoiceScorer(use_letter_onehot=use_letter_onehot).fit(rows)
    return model, len(mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--dataset-zip", type=str, default=None)
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--output-dir", type=str, default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(args.output_dir, exist_ok=True)

    law_items, examples = load_corpus(args.dataset_root, args.dataset_zip)
    print(f"Loaded {len(law_items)} law articles, {len(examples)} labeled train examples")
    if not law_items:
        raise SystemExit("law_items rong - kiem tra lai schema/duong dan law_db")

    law_lookup = build_law_lookup(law_items)
    top_k = cfg["retriever"]["top_k"]
    max_law_chars = cfg["retriever"]["max_law_chars"]
    retriever = build_retriever(law_items, cfg["retriever"]["method"], top_k, max_law_chars)
    content_vectorizer = build_content_vectorizer(law_items, max_law_chars=max_law_chars)

    examples, dropped_ids = clean_train_examples(examples)
    print(f"Dropped {len(dropped_ids)} example(s) with invalid MC answer: {dropped_ids}")

    val_size = cfg["model"]["val_size"]
    seed = cfg["model"]["seed"]
    train_ex, val_ex = split_examples(examples, val_size=val_size, seed=seed)
    print(f"Split: {len(train_ex)} train / {len(val_ex)} val (seed={seed})")

    yesno_model, n_yn = train_yesno(train_ex)
    print(f"Trained YesNoClassifier on {n_yn} train examples")

    mc_model, n_mc = train_mc(train_ex, retriever, law_lookup, content_vectorizer, False, top_k, max_law_chars)
    print(f"Trained ChoiceScorer (no letter one-hot) on {n_mc} train examples")

    mc_model_letter, _ = train_mc(train_ex, retriever, law_lookup, content_vectorizer, True, top_k, max_law_chars)
    print("Trained ChoiceScorer (with letter one-hot, ablation)")

    yesno_model.save(os.path.join(args.output_dir, "yesno.joblib"))
    mc_model.save(os.path.join(args.output_dir, "mc_choice.joblib"))
    mc_model_letter.save(os.path.join(args.output_dir, "mc_choice_with_letter.joblib"))

    import joblib
    joblib.dump({"retriever": retriever, "content_vectorizer": content_vectorizer, "law_items": law_items},
                os.path.join(args.output_dir, "retrieval.joblib"))

    split_info = {
        "seed": seed,
        "val_size": val_size,
        "train_ids": [ex.get("id") for ex in train_ex],
        "val_ids": [ex.get("id") for ex in val_ex],
        "dropped_ids": dropped_ids,
        "retriever_method": cfg["retriever"]["method"],
    }
    with open(os.path.join(args.output_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"Done. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
