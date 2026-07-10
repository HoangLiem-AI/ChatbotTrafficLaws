import argparse
import json
import os
import sys

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_from_root, load_from_zip, flatten_laws
from src.retriever_tfidf import TFIDFRetriever
from src.retriever_bm25 import BM25Retriever
from src.retrieval_eval import evaluate_retriever
from src.qa_model import normalize_label, build_law_lookup, YesNoClassifier, ChoiceScorer
from src.qa_rules import answer as rule_based_answer
from scripts.train import clean_train_examples


def load_corpus(dataset_root=None, dataset_zip=None):
    if dataset_root:
        laws, examples = load_from_root(dataset_root, split="train")
    elif dataset_zip:
        laws, examples = load_from_zip(dataset_zip, split="train")
    else:
        raise SystemExit("Cần --dataset-root hoặc --dataset-zip")
    return flatten_laws(laws), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--dataset-zip", type=str, default=None)
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--models-dir", type=str, default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    top_k = cfg["retriever"]["top_k"]
    max_law_chars = cfg["retriever"]["max_law_chars"]

    law_items, examples = load_corpus(args.dataset_root, args.dataset_zip)
    law_lookup = build_law_lookup(law_items)
    examples, _ = clean_train_examples(examples)

    with open(os.path.join(args.models_dir, "split.json"), encoding="utf-8") as f:
        split_info = json.load(f)
    val_ids = set(split_info["val_ids"])
    val_ex = [ex for ex in examples if ex.get("id") in val_ids]
    print(f"Loaded val split: {len(val_ex)} examples (seed={split_info['seed']})")

    # --- Bang 1: Retrieval ---
    tfidf_retriever = TFIDFRetriever(law_items, max_law_chars=max_law_chars)
    bm25_retriever = BM25Retriever(law_items, max_law_chars=max_law_chars)
    retrieval_rows = []
    for name, retriever in [("TF-IDF", tfidf_retriever), ("BM25", bm25_retriever)]:
        metrics = evaluate_retriever(retriever, val_ex, ks=(1, 3, 5))
        retrieval_rows.append({"method": name, **metrics})
    print("\n=== Bang 1: Retrieval (Recall@k, MRR) tren val ===")
    print(pd.DataFrame(retrieval_rows).to_string(index=False))

    # --- Bang 2: Yes/No ---
    yn_val = [ex for ex in val_ex if (ex.get("question_type") or "").strip().lower() == "yes/no"]
    yn_gold = [normalize_label(ex["answer"]) for ex in yn_val]
    yn_questions = [ex["question"] for ex in yn_val]

    yn_baseline_pred = [
        rule_based_answer(q, "Yes/No", yes_keywords=cfg["qa_rules"]["yes_keywords"], no_keywords=cfg["qa_rules"]["no_keywords"])
        for q in yn_questions
    ]
    yesno_model = YesNoClassifier.load(os.path.join(args.models_dir, "yesno.joblib"))
    yn_model_pred = yesno_model.predict(yn_questions)

    yn_rows = []
    for name, preds in [("Baseline (keyword rules)", yn_baseline_pred), ("YesNoClassifier (LogReg)", yn_model_pred)]:
        preds = [normalize_label(p) for p in preds]
        yn_rows.append({
            "method": name,
            "accuracy": accuracy_score(yn_gold, preds),
            "macro_f1": f1_score(yn_gold, preds, average="macro", zero_division=0),
        })
    print(f"\n=== Bang 2: Yes/No ({len(yn_val)} vi du val) ===")
    print(pd.DataFrame(yn_rows).to_string(index=False))

    # --- Bang 3: Multiple choice ---
    mc_val = [ex for ex in val_ex if (ex.get("question_type") or "").strip().lower().startswith("multiple")]
    mc_gold = [normalize_label(ex["answer"]).upper() for ex in mc_val]

    mc_baseline_pred = [rule_based_answer(ex["question"], "Multiple choice") for ex in mc_val]

    content_vec_bundle = __import__("joblib").load(os.path.join(args.models_dir, "retrieval.joblib"))
    content_vectorizer = content_vec_bundle["content_vectorizer"]
    saved_retriever = content_vec_bundle["retriever"]

    mc_model = ChoiceScorer.load(os.path.join(args.models_dir, "mc_choice.joblib"))
    mc_model_letter = ChoiceScorer.load(os.path.join(args.models_dir, "mc_choice_with_letter.joblib"))
    mc_model_pred = [mc_model.predict_example(ex, saved_retriever, law_lookup, content_vectorizer, top_k=top_k, max_ctx_chars=max_law_chars) for ex in mc_val]
    mc_model_letter_pred = [mc_model_letter.predict_example(ex, saved_retriever, law_lookup, content_vectorizer, top_k=top_k, max_ctx_chars=max_law_chars) for ex in mc_val]

    mc_rows = []
    for name, preds in [
        ("Baseline (MD5 hash)", mc_baseline_pred),
        ("ChoiceScorer (no letter feature)", mc_model_pred),
        ("ChoiceScorer (+letter one-hot, ablation)", mc_model_letter_pred),
    ]:
        preds = [normalize_label(p).upper() for p in preds]
        mc_rows.append({"method": name, "accuracy": accuracy_score(mc_gold, preds)})
    print(f"\n=== Bang 3: Multiple choice ({len(mc_val)} vi du val) ===")
    print(pd.DataFrame(mc_rows).to_string(index=False))


if __name__ == "__main__":
    main()
