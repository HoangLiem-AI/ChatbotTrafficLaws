import argparse, os, json, zipfile, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_laws_from_root, load_laws_from_zip, load_examples_from_root, load_examples_from_zip, flatten_laws
from src.qa_model import build_law_lookup, YesNoClassifier, ChoiceScorer


def load_models(models_dir):
    import joblib
    yesno_model = YesNoClassifier.load(os.path.join(models_dir, "yesno.joblib"))
    mc_model = ChoiceScorer.load(os.path.join(models_dir, "mc_choice.joblib"))
    bundle = joblib.load(os.path.join(models_dir, "retrieval.joblib"))
    return yesno_model, mc_model, bundle["retriever"], bundle["content_vectorizer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", type=str, default=None)
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--output-dir", type=str, default="outputs")
    ap.add_argument("--split", type=str, choices=["train", "public_task1", "public_task2"], default="public_task1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    top_k = cfg["retriever"]["top_k"]
    max_law_chars = cfg["retriever"]["max_law_chars"]

    if args.dataset_root:
        laws = load_laws_from_root(args.dataset_root)
        data = load_examples_from_root(args.dataset_root, args.split)
    elif args.dataset_zip:
        laws = load_laws_from_zip(args.dataset_zip)
        data = load_examples_from_zip(args.dataset_zip, args.split)
    else:
        raise SystemExit("Cần --dataset-zip hoặc --dataset-root")

    law_items = flatten_laws(laws)
    law_lookup = build_law_lookup(law_items)
    yesno_model, mc_model, retriever, content_vectorizer = load_models(args.models_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    t1, t2 = [], []

    write_task1 = args.split in ("train", "public_task1")
    write_task2 = args.split in ("train", "public_task2")

    for ex in data:
        q = ex.get("question", "")
        qtype = ex.get("question_type", "")

        if write_task1:
            topk = retriever.search(q, top_k=top_k) or [{"law_id": "Không xác định", "article_id": "0", "score": 0.0}]
            t1.append({"id": ex.get("id", ""), "image_id": ex.get("image_id", ""), "question": q, "relevant_articles": topk})

        if write_task2:
            qtype_norm = (qtype or "Yes/No").strip().lower()
            if qtype_norm.startswith("multiple"):
                ans = mc_model.predict_example(ex, retriever, law_lookup, content_vectorizer, top_k=top_k, max_ctx_chars=max_law_chars)
            else:
                ans = yesno_model.predict([q])[0]
            ctx_articles = ex.get("relevant_articles") or retriever.search(q, top_k=top_k)
            t2.append({"id": ex.get("id", ""), "image_id": ex.get("image_id", ""), "question": q,
                       "question_type": qtype or "Yes/No", "relevant_articles": ctx_articles, "answer": ans})

    if write_task1:
        json.dump(t1, open(os.path.join(args.output_dir, "submission_task1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if write_task2:
        json.dump(t2, open(os.path.join(args.output_dir, "submission_task2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with zipfile.ZipFile(os.path.join(args.output_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        if write_task1:
            z.write(os.path.join(args.output_dir, "submission_task1.json"), arcname="submission_task1.json")
        if write_task2:
            z.write(os.path.join(args.output_dir, "submission_task2.json"), arcname="submission_task2.json")

    print("Done. See:", args.output_dir)


if __name__ == "__main__":
    main()
