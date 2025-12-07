import argparse, csv, json, os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from embedder import embed_text, tokenize
    from model_io import load_resources, predict_proba_array
except Exception:
    from src.embedder import embed_text, tokenize  
    from src.model_io import load_resources, predict_proba_array  

# Heurística opcional
try:
    from heuristics import adjust_probability_with_heuristics
except Exception:
    adjust_probability_with_heuristics = None

def read_labeled_csv(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "texto" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("CSV deve ter cabeçalhos: texto,label (label 0/1).")
        for row in reader:
            t = (row.get("texto") or "").strip()
            if t == "":
                continue
            y = int(row.get("label"))
            texts.append(t)
            labels.append(y)
    return texts, np.array(labels, dtype=int)

def main():
    ap = argparse.ArgumentParser("Sintoniza threshold para maximizar uma métrica")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--strip_accents", action="store_true")
    ap.add_argument("--use_heuristics", action="store_true")
    ap.add_argument("--metric", default="precision_at_recall", choices=["f1","precision","recall","accuracy","precision_at_recall"])
    ap.add_argument("--recall_min", type=float, default=0.70, help="Usado com metric=precision_at_recall")
    ap.add_argument("--t_min", type=float, default=0.30)
    ap.add_argument("--t_max", type=float, default=0.85)
    ap.add_argument("--t_steps", type=int, default=111)
    args = ap.parse_args()

    model, scaler, word_map, saved_th, model_type = load_resources(args.model_dir, args.strip_accents)
    texts, y_true = read_labeled_csv(args.input_csv)

    X, toks_list = [], []
    for t in texts:
        toks = tokenize(t, lowercase=True, strip_accents=args.strip_accents)
        toks_list.append(toks)
        X.append(embed_text(t, word_map, lowercase=True, strip_accents=args.strip_accents))
    X = np.stack(X, axis=0)

    proba = predict_proba_array(model, scaler.transform(X), model_type).astype(float)
    if args.use_heuristics and adjust_probability_with_heuristics is not None:
        proba = np.array([adjust_probability_with_heuristics(float(p), toks) for p, toks in zip(proba, toks_list)], dtype=float)

    best = None
    for th in np.linspace(args.t_min, args.t_max, args.t_steps):
        y_pred = (proba >= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        if args.metric == "precision":
            score = prec
        elif args.metric == "recall":
            score = rec
        elif args.metric == "accuracy":
            score = acc
        elif args.metric == "precision_at_recall":
            score = prec if rec >= args.recall_min else -1.0
        else:
            score = f1

        row = {"threshold": float(th), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "score": float(score)}
        if best is None or score > best["score"]:
            best = row

    print(json.dumps({
        "saved_threshold": float(saved_th),
        "metric": args.metric,
        "recall_min": args.recall_min if args.metric == "precision_at_recall" else None,
        "best": best
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()