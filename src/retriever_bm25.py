import re
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, law_items, max_law_chars=1200):
        self.items = law_items
        self.tokenized = [_tokenize((x["content"] or "")[:max_law_chars]) for x in self.items]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, question, top_k=3):
        scores = self.bm25.get_scores(_tokenize(question))
        idx = np.argsort(-scores)[:top_k]
        out = []
        for i in idx:
            it = self.items[int(i)]
            out.append({
                "law_id": it["law_id"],
                "article_id": it["article_id"],
                "score": float(scores[int(i)]),
            })
        return out
