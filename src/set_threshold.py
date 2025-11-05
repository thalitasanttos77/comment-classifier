import os
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser("Define permanentemente o decision_threshold dentro do model.pkl")
    parser.add_argument("--model_dir", type=str, default="models", help="Pasta com model.pkl")
    parser.add_argument("--value", type=float, required=True, help="Novo threshold (ex.: 0.60)")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise SystemExit(f"Arquivo não encontrado: {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    old = float(bundle.get("decision_threshold", 0.5))
    bundle["decision_threshold"] = float(args.value)

    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"decision_threshold atualizado de {old:.2f} para {args.value:.2f} em {os.path.abspath(model_path)}")

if __name__ == "__main__":
    main()