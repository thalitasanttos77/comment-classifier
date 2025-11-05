import os
import json
import pickle
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)
from data_io import load_all

def fit_logreg(X_tr, y_tr, C, calibrate: str | None = None):
    """
    Treina Regressão Logística com StandardScaler.
    Se calibrate in {"sigmoid","isotonic"}: usa CalibratedClassifierCV(cv=5) em X_tr já escalonado.
    Retorna (scaler, clf_ou_clf_calibrado).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    base = LogisticRegression(max_iter=2000, class_weight="balanced", C=C, solver="lbfgs")
    if calibrate in { "sigmoid", "isotonic" }:
        clf = CalibratedClassifierCV(base, method=calibrate, cv=5)
    else:
        clf = base
    clf.fit(X_tr_s, y_tr)
    return scaler, clf

def search_C(X, y, seed):
    """
    Seleciona C via CV (sem calibração aqui para reduzir custo).
    Métrica: F1 com limiar 0.5 na predição.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    Cs = [0.1, 0.3, 1.0, 3.0, 10.0]
    best_C, best_f1 = None, -1.0
    for C in Cs:
        f1s = []
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            scaler, clf = fit_logreg(X_tr, y_tr, C, calibrate=None)
            X_va_s = scaler.transform(X_va)
            proba = clf.predict_proba(X_va_s)[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            f1s.append(f1_score(y_va, y_pred))
        avg_f1 = float(np.mean(f1s))
        if avg_f1 > best_f1:
            best_f1, best_C = avg_f1, C
    return best_C, best_f1

def search_threshold(clf, scaler, X_va, y_va):
    """
    Varre thresholds e retorna o que maximiza F1 na validação.
    """
    X_va_s = scaler.transform(X_va)
    proba = clf.predict_proba(X_va_s)[:, 1]
    thresholds = np.linspace(0.3, 0.7, 41)  # 0.30 a 0.70 de 0.01
    best_th, best_f1 = 0.5, -1.0
    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        f1 = f1_score(y_va, y_pred)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1

def evaluate(clf, scaler, X_te, y_te, threshold):
    X_te_s = scaler.transform(X_te)
    proba = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_te, y_pred).tolist()
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dados")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate", type=str, default="sigmoid", choices=["none","sigmoid","isotonic"], help="Calibração de probabilidades")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    words, word_vecs, X, y = load_all(args.data_dir)
    X = X.astype(np.float32)
    y = y.astype(int)

    # Split: 70/15/15 (train/val/test)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=args.seed, stratify=y_tmp)
    # 0.17647 ~ 15% do total (0.85 * 0.17647 ~ 0.15)

    # 1) Seleciona melhor C (sem calibração para velocidade)
    best_C, cv_f1 = search_C(X_tr, y_tr, seed=args.seed)

    # 2) Treina no treino e ajusta threshold na validação (com calibração, se pedida)
    calibrate = None if args.calibrate == "none" else args.calibrate
    scaler, clf = fit_logreg(X_tr, y_tr, best_C, calibrate=calibrate)
    best_th, va_f1 = search_threshold(clf, scaler, X_va, y_va)

    # 3) Re-treina em treino+val com best_C (e calibração opcional) e avalia no teste com threshold escolhido
    X_trva = np.vstack([X_tr, X_va])
    y_trva = np.concatenate([y_tr, y_va])
    scaler_final, clf_final = fit_logreg(X_trva, y_trva, best_C, calibrate=calibrate)
    test_metrics = evaluate(clf_final, scaler_final, X_te, y_te, threshold=best_th)

    # Salvar modelo + threshold escolhido
    bundle = {
        "scaler": scaler_final,
        "model": clf_final,
        "decision_threshold": float(best_th),
        "model_type": "logreg",
        "C": float(best_C),
        "calibration": args.calibrate,
    }
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    # Recursos de vocabulário para a inferência (item b/c)
    np.save(os.path.join(args.out_dir, "words.npy"), np.array(words, dtype=object))
    np.save(os.path.join(args.out_dir, "word_vectors.npy"), word_vecs.astype(np.float32))

    metrics = {
        "cv_best_C": best_C,
        "cv_mean_f1": cv_f1,
        "val_best_threshold": best_th,
        "val_best_f1": va_f1,
        "calibration": args.calibrate,
        "test": test_metrics,
        "sizes": {
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_va.shape[0]),
            "n_test": int(X_te.shape[0]),
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Treinamento+validação concluídos.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Modelo salvo em:", os.path.abspath(os.path.join(args.out_dir, "model.pkl")))

if __name__ == "__main__":
    main()