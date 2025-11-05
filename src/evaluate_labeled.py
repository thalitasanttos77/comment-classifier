import os
import csv
import json
import argparse
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from heuristics import adjust_probability_with_heuristics

# importa tanto via execução da raiz (-m) quanto de dentro de src
try:
    from src.embedder import build_word_map, embed_text, tokenize
except ImportError:
    from embedder import build_word_map, embed_text, tokenize

def load_resources(model_dir: str, strip_accents: bool):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    words = np.load(os.path.join(model_dir, "words.npy"), allow_pickle=True).tolist()
    word_vectors = np.load(os.path.join(model_dir, "word_vectors.npy"))
    word_map = build_word_map(words, word_vectors, lowercase=True, strip_accents=strip_accents)
    threshold = float(bundle.get("decision_threshold", 0.5))
    return bundle["model"], bundle["scaler"], word_map, threshold

def read_labeled_csv(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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

    model, scaler, word_map, saved_th = load_resources(args.model_dir, args.strip_accents)
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

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_s)[:, 1].astype(float)
        if args.use_heuristics:
            proba = np.array([adjust_probability_with_heuristics(p, toks) for p, toks in zip(proba, toks_list)], dtype=float)
        y_pred = (proba >= decision_th).astype(int)
    else:
        y_pred = model.predict(X_s).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, digits=4)

    metrics = {
        "n": int(len(y_true)),
        "threshold": decision_th,
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