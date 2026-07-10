from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFRetriever:
    def __init__(self, law_items, max_law_chars=1200):
        self.items = law_items
        texts = [(x["content"] or "")[:max_law_chars] for x in self.items]
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self.mat = self.vectorizer.fit_transform(texts)

    def search(self, question, top_k=3):
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.mat).ravel()
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            it = self.items[int(i)]
            out.append({
                "law_id": it["law_id"],
                "article_id": it["article_id"],
                "score": float(sims[int(i)])
            })
        return out
