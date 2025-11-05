import os
import sys
import pickle
import argparse
import numpy as np
from embedder import build_word_map, embed_text, tokenize

# Heurística opcional (se não existir, segue sem)
try:
    from heuristics import adjust_probability_with_heuristics
except Exception:
    adjust_probability_with_heuristics = None

def load_resources(model_dir: str, strip_accents: bool):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    words = np.load(os.path.join(model_dir, "words.npy"), allow_pickle=True).tolist()
    word_vectors = np.load(os.path.join(model_dir, "word_vectors.npy"))
    word_map = build_word_map(words, word_vectors, lowercase=True, strip_accents=strip_accents)
    threshold = float(bundle.get("decision_threshold", 0.5))
    return bundle["model"], bundle["scaler"], word_map, threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models", help="Pasta com model.pkl, words.npy, word_vectors.npy")
    parser.add_argument("--strip_accents", action="store_true", help="Remover acentos ao tokenizar")
    parser.add_argument("--threshold", type=float, default=None, help="Se não passar, usa o salvo no modelo (decision_threshold)")
    parser.add_argument("--use_heuristics", action="store_true", help="Aplicar regras simples (negação/léxico/expressões)")
    args = parser.parse_args()

    model, scaler, word_map, saved_th = load_resources(args.model_dir, args.strip_accents)
    decision_th = float(args.threshold) if args.threshold is not None else saved_th

    print("Digite um comentário (Ctrl+C para sair):")
    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue

            toks = tokenize(text, lowercase=True, strip_accents=args.strip_accents)
            x = embed_text(text, word_map, lowercase=True, strip_accents=args.strip_accents).reshape(1, -1)
            if np.allclose(x, 0.0):
                print("Aviso: nenhuma palavra conhecida no texto. Resultado pode ser pouco confiável.")

            x_s = scaler.transform(x)

            if hasattr(model, "predict_proba"):
                prob_pos = float(model.predict_proba(x_s)[0, 1])
                if args.use_heuristics and adjust_probability_with_heuristics is not None:
                    prob_pos = adjust_probability_with_heuristics(prob_pos, toks)
                pred = 1 if prob_pos >= decision_th else 0
                label = "positivo" if pred == 1 else "negativo"
                conf = prob_pos if pred == 1 else (1.0 - prob_pos)
                conf_pct = round(conf * 100)
                print(f"Classificação: Este comentário é {label}")
                print(f"Probabilidade de ser {label}: {conf_pct}%\n")
            else:
                pred = int(model.predict(x_s)[0])
                label = "positivo" if pred == 1 else "negativo"
                print(f"Classificação: Este comentário é {label}")
                print("Probabilidade: não disponível para este modelo.\n")
    except KeyboardInterrupt:
        print("\nEncerrado.")
        sys.exit(0)

if __name__ == "__main__":
    main()