import os
import csv
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from embedder import embed_text, tokenize
from model_io import load_resources, predict_proba_array

# Heurística opcional
try:
    from heuristics import adjust_probability_with_heuristics
except Exception:
    adjust_probability_with_heuristics = None

def read_labeled_csv(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV deve ter cabeçalhos: texto,label")
        if "texto" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("CSV deve ter cabeçalhos: texto,label (com label 0/1).")
        for row in reader:
            t = (row.get("texto") or "").strip()
            if t == "":
                continue
            y = int(row.get("label"))
            texts.append(t)
            labels.append(y)
    return texts, np.array(labels, dtype=int)

def main():
    parser = argparse.ArgumentParser("Avaliador de lote rotulado (CSV: texto,label).")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV UTF-8 com colunas: texto,label")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Se ausente, usa o salvo no modelo.")
    parser.add_argument("--use_heuristics", action="store_true", help="Ajustar probabilidade com regras simples")
    args = parser.parse_args()

    model, scaler, word_map, saved_th, model_type = load_resources(args.model_dir, args.strip_accents)
    decision_th = float(args.threshold) if args.threshold is not None else saved_th

    texts, y_true = read_labeled_csv(args.input_csv)

    X = []
    toks_list = []
    for t in texts:
        toks = tokenize(t, lowercase=True, strip_accents=args.strip_accents)
        toks_list.append(toks)
        X.append(embed_text(t, word_map, lowercase=True, strip_accents=args.strip_accents))
    X = np.stack(X, axis=0)
    X_s = scaler.transform(X)

    proba = predict_proba_array(model, X_s, model_type).astype(float)
    if args.use_heuristics and adjust_probability_with_heuristics is not None:
        proba = np.array([adjust_probability_with_heuristics(float(p), toks) for p, toks in zip(proba, toks_list)], dtype=float)
    y_pred = (proba >= decision_th).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, digits=4)

    metrics = {
        "n": int(len(y_true)),
        "threshold": float(decision_th),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": rep,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()