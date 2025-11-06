import os
import sys
import argparse
import numpy as np

from embedder import embed_text, tokenize
from model_io import load_resources, predict_proba_array

# Heurística opcional
try:
    from heuristics import adjust_probability_with_heuristics
except Exception:
    adjust_probability_with_heuristics = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models", help="Pasta com model.pkl, words.npy, word_vectors.npy")
    parser.add_argument("--strip_accents", action="store_true", help="Remover acentos ao tokenizar")
    parser.add_argument("--threshold", type=float, default=None, help="Se não passar, usa o salvo no modelo")
    parser.add_argument("--use_heuristics", action="store_true", help="Aplicar regras simples (negação/léxico)")
    args = parser.parse_args()

    model, scaler, word_map, saved_th, model_type = load_resources(args.model_dir, args.strip_accents)
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
            proba = float(predict_proba_array(model, x_s, model_type).reshape(-1)[0])

            if args.use_heuristics and adjust_probability_with_heuristics is not None:
                proba = adjust_probability_with_heuristics(proba, toks)

            pred = 1 if proba >= decision_th else 0
            label = "positivo" if pred == 1 else "negativo"
            conf = proba if pred == 1 else (1.0 - proba)
            conf_pct = round(conf * 100)
            print(f"Classificação: Este comentário é {label}")
            print(f"Probabilidade de ser {label}: {conf_pct}%\n")
    except KeyboardInterrupt:
        print("\nEncerrado.")
        sys.exit(0)

if __name__ == "__main__":
    main()