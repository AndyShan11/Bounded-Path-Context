"""KGQA evaluation: Hits@1, F1, precision, recall."""


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def hits1(predicted, gold):
    if not predicted or not gold:
        return 0.0
    p0 = _norm(predicted[0])
    return float(any(p0 == _norm(g) for g in gold))


def precision_recall_f1(predicted, gold):
    if not gold:
        return 0.0, 0.0, 0.0
    P = {_norm(p) for p in predicted}
    G = {_norm(g) for g in gold}
    tp = len(P & G)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1
