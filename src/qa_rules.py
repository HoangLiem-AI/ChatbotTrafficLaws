import hashlib

def _stable_choice_hash(text):
    h = hashlib.md5(text.encode('utf-8')).hexdigest()
    idx = int(h, 16) % 4
    return ["A","B","C","D"][idx]

def answer(question, question_type, choices=None, yes_keywords=None, no_keywords=None):
    qtype = (question_type or "").strip().lower()
    if qtype.startswith("multiple"):
        return _stable_choice_hash(question or "")
    q = (question or "").lower()
    yes_kw = [k.lower() for k in (yes_keywords or [])]
    no_kw  = [k.lower() for k in (no_keywords or [])]
    if any(k in q for k in yes_kw) and not any(k in q for k in no_kw):
        return "Đúng"
    if any(k in q for k in no_kw) and not any(k in q for k in yes_kw):
        return "Sai"
    return "Đúng"
