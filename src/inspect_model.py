import os
import argparse
import pickle
import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:
    pass

def main():
    p = argparse.ArgumentParser("Inspeciona model.pkl e mostra o que está salvo/ativo")
    p.add_argument("--model_dir", type=str, default="models", help="Pasta que contém model.pkl, words.npy, word_vectors.npy")
    args = p.parse_args()

    model_path = os.path.join(args.model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise SystemExit(f"Não encontrado: {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print("Chaves em model.pkl:", list(bundle.keys()))

    decision_threshold = bundle.get("decision_threshold", 0.5)
    model_type = bundle.get("model_type", type(bundle.get("model")).__name__)
    C = bundle.get("C", None)
    calibration = bundle.get("calibration", None)

    print(f"decision_threshold: {decision_threshold}")
    print(f"model_type: {model_type}")
    print(f"C: {C}")
    print(f"calibration: {calibration}")

    model = bundle.get("model")
    scaler = bundle.get("scaler")
    pca = bundle.get("pca", None)  

    print("\n[Modelo]")
    print("classe:", type(model).__name__)
    print("tem predict_proba?:", hasattr(model, "predict_proba"))

    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(model, CalibratedClassifierCV):
            print("CalibratedClassifierCV.method:", getattr(model, "method", "desconhecido"))
            base = getattr(model, "base_estimator", None)
            print("base_estimator:", type(base).__name__ if base is not None else None)
            if hasattr(base, "get_params"):
                params = base.get_params()
                for k in ["C", "class_weight", "solver", "max_iter"]:
                    if k in params:
                        print(f"base_estimator.{k}:", params[k])
        else:
            try:
                from sklearn.linear_model import LogisticRegression
                if isinstance(model, LogisticRegression):
                    params = model.get_params()
                    for k in ["C", "class_weight", "solver", "max_iter"]:
                        print(f"LogisticRegression.{k}:", params.get(k))
            except Exception:
                pass
    except Exception:
        pass

    print("\n[Scaler]")
    print("classe:", type(scaler).__name__ if scaler is not None else None)
    try:
        if scaler is not None and hasattr(scaler, "mean_"):
            print("mean_.shape:", scaler.mean_.shape)
            print("scale_.shape:", scaler.scale_.shape)
    except Exception:
        pass

    if pca is not None:
        print("\n[PCA]")
        print("classe:", type(pca).__name__)
        try:
            ev = getattr(pca, "explained_variance_ratio_", None)
            if ev is not None:
                print("variância explicada total:", float(np.sum(ev)))
                print("n_componentes:", len(ev))
        except Exception:
            pass
    else:
        print("\n[PCA] não presente (pipeline sem PCA).")

    words_path = os.path.join(args.model_dir, "words.npy")
    if os.path.exists(words_path):
        words = np.load(words_path, allow_pickle=True)
        print(f"\nVocabulário: {len(words)} tokens em words.npy")
    else:
        print("\nwords.npy não encontrado.")

    print("\nDica: inferência e planilha podem usar automaticamente o decision_threshold salvo; se você passar --threshold, ele sobrescreve o valor salvo.")

if __name__ == "__main__":
    main()