import os, json, zipfile

LAW_FILENAMES = ["vlsp2025_law_new.json", "vlsp2025_law.json"]
SPLIT_FILES = {
    "train": ("train_data", "vlsp_2025_train.json"),
    "public_task1": ("public_test", "vlsp_2025_public_test_task1.json"),
    "public_task2": ("public_test", "vlsp_2025_public_test_task2.json"),
}


def _detect_prefix(names):
    for n in names:
        if n.endswith("README.txt"):
            return n[:-10]
    import os as _os
    return _os.path.commonprefix(names)


def load_laws_from_root(root):
    for fname in LAW_FILENAMES:
        path = os.path.join(root, "law_db", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Không tìm thấy file luật trong {os.path.join(root, 'law_db')} (đã thử: {LAW_FILENAMES})")


def load_examples_from_root(root, split="train"):
    subdir, fname = SPLIT_FILES[split]
    path = os.path.join(root, subdir, fname)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_laws_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        prefix = _detect_prefix(names)
        for fname in LAW_FILENAMES:
            candidate = prefix + "law_db/" + fname
            if candidate in names:
                return json.loads(z.read(candidate).decode("utf-8"))
    raise FileNotFoundError(f"Không tìm thấy file luật trong ZIP (đã thử: {LAW_FILENAMES})")


def load_examples_from_zip(zip_path, split="train"):
    subdir, fname = SPLIT_FILES[split]
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        prefix = _detect_prefix(names)
        return json.loads(z.read(prefix + f"{subdir}/{fname}").decode("utf-8"))


def load_from_root(root, split="train"):
    return load_laws_from_root(root), load_examples_from_root(root, split)


def load_from_zip(zip_path, split="train"):
    return load_laws_from_zip(zip_path), load_examples_from_zip(zip_path, split)


def flatten_laws(laws):
    items = []

    def add(law_id, article_id, content):
        if content and isinstance(content, str):
            items.append({"law_id": law_id, "article_id": str(article_id), "content": content})

    if isinstance(laws, list) and laws and isinstance(laws[0], dict) and "articles" in laws[0]:
        # Schema thật VLSP 2025: [{"id"/"law_id", "title", "articles": [{"id","title","text"}]}]
        for law in laws:
            law_id = law.get("id") or law.get("law_id") or law.get("law") or "UNKNOWN_LAW"
            for art in law.get("articles", []):
                article_id = art.get("id") or art.get("article_id") or "UNKNOWN_ART"
                title = (art.get("title") or "").strip()
                text = (art.get("text") or art.get("content") or "").strip()
                content = (title + "\n" + text).strip()
                add(law_id, article_id, content)
    elif isinstance(laws, list):
        for x in laws:
            law_id = x.get("law_id") or x.get("law") or "UNKNOWN_LAW"
            article_id = x.get("article_id") or x.get("article") or x.get("id") or "UNKNOWN_ART"
            content = x.get("content") or x.get("text") or ""
            add(law_id, article_id, content)
    elif isinstance(laws, dict):
        for law_id, group in laws.items():
            if isinstance(group, dict):
                for article_id, content in group.items():
                    if isinstance(content, str):
                        add(law_id, str(article_id), content)
                    else:
                        import json as _json
                        add(law_id, str(article_id), _json.dumps(content, ensure_ascii=False))
    return items
