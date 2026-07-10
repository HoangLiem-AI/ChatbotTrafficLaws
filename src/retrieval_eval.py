def evaluate_retriever(retriever, examples, ks=(1, 3, 5)):
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n = 0
    for ex in examples:
        gold = {(a.get("law_id"), str(a.get("article_id"))) for a in ex.get("relevant_articles", [])}
        if not gold:
            continue
        n += 1
        ranked = retriever.search(ex.get("question", ""), top_k=max_k)
        ranked_keys = [(r["law_id"], r["article_id"]) for r in ranked]
        rank = next((i + 1 for i, key in enumerate(ranked_keys) if key in gold), None)
        if rank is not None:
            rr_sum += 1.0 / rank
        for k in ks:
            if any(key in gold for key in ranked_keys[:k]):
                hits[k] += 1
    if n == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    result = {f"recall@{k}": hits[k] / n for k in ks}
    result["mrr"] = rr_sum / n
    return result
