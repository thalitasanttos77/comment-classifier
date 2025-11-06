import os
import json
import pickle
import argparse
import numpy as np
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow import keras

try:
    from data_io import load_all
except Exception:
    from src.data_io import load_all  

def build_mlp(input_dim: int, hidden=(128, 64), dropout=0.2, l2=1e-4, lr=1e-3, seed=42) -> keras.Model:
    tf.keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    inp = keras.Input(shape=(input_dim,), name="x")
    x = inp
    for h in hidden:
        x = keras.layers.Dense(h, activation="relu", kernel_regularizer=reg)(x)
        if dropout and dropout > 0:
            x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="prob_pos")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model

def fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, class_weight: dict, seed=42) -> keras.Model:
    model = build_mlp(X_tr.shape[1], hidden=(128, 64), dropout=0.25, l2=1e-4, lr=1e-3, seed=seed)
    es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=120,
        batch_size=128,
        verbose=0,
        class_weight=class_weight,
        callbacks=[es],
    )
    return model

def pick_threshold(proba: np.ndarray, y_va: np.ndarray, metric: str, recall_min: float):
    best_th, best_score, best_row = 0.5, -1.0, None
    for th in np.linspace(0.30, 0.85, 111):
        y_pred = (proba >= th).astype(int)
        acc = accuracy_score(y_va, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_va, y_pred, average="binary", zero_division=0)
        if metric == "precision":
            score = prec
        elif metric == "recall":
            score = rec
        elif metric == "accuracy":
            score = acc
        elif metric == "precision_at_recall":
            score = prec if rec >= recall_min else -1.0
        else:
            score = f1
        if score > best_score:
            best_score, best_th = float(score), float(th)
            best_row = {"threshold": best_th, "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return best_th, best_row

def evaluate(model: keras.Model, X_te: np.ndarray, y_te: np.ndarray, threshold: float) -> Dict[str, float]:
    proba = model.predict(X_te, verbose=0).reshape(-1)
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_te, y_pred).tolist()
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm}

def main():
    ap = argparse.ArgumentParser("Treino MLP (TensorFlow) sobre WTEXpc/CLtx com foco em precisão")
    ap.add_argument("--data_dir", type=str, default="dados")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", type=str, default="precision_at_recall", choices=["f1","precision","recall","accuracy","precision_at_recall"], help="Métrica para escolher o threshold na validação")
    ap.add_argument("--recall_min", type=float, default=0.70, help="Usado quando metric=precision_at_recall")
    ap.add_argument("--neg_bias", type=float, default=1.5, help="Multiplica o peso da classe 0 para reduzir FPs (ex.: 1.5~2.0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    words, word_vecs, X, y = load_all(args.data_dir)
    X = X.astype(np.float32)
    y = y.astype(int)

    # Split 70/15/15
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=args.seed, stratify=y_tmp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Pesos de classe (balanceados) + viés negativo para punir FPs
    classes = np.unique(y_tr)
    cw_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_values)}
    class_weight[0] = class_weight.get(0, 1.0) * float(args.neg_bias)

    # Treina e escolhe limiar por métrica
    model = fit_mlp(X_tr_s, y_tr, X_va_s, y_va, class_weight=class_weight, seed=args.seed)
    proba_va = model.predict(X_va_s, verbose=0).reshape(-1)
    best_th, best_row = pick_threshold(proba_va, y_va, metric=args.metric, recall_min=args.recall_min)

    # Re-treina em treino+val (usa mesmo class_weight) e avalia no teste
    X_trva_s = np.vstack([X_tr_s, X_va_s]); y_trva = np.concatenate([y_tr, y_va])
    model_f = fit_mlp(X_trva_s, y_trva, X_va_s, y_va, class_weight=class_weight, seed=args.seed)  # usa val para early stopping
    test_metrics = evaluate(model_f, X_te_s, y_te, threshold=best_th)

    # Salva Keras + bundle
    keras_path = os.path.join(args.out_dir, "tf_model.keras")
    model_f.save(keras_path)
    bundle = {
        "scaler": scaler,
        "model_type": "tf_keras",
        "model_path": "tf_model.keras",
        "decision_threshold": float(best_th),
        "threshold_selection": {"metric": args.metric, "recall_min": args.recall_min, "val_metrics_at_best": best_row}
    }
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    # Vocabulário
    np.save(os.path.join(args.out_dir, "words.npy"), np.array(words, dtype=object))
    np.save(os.path.join(args.out_dir, "word_vectors.npy"), word_vecs.astype(np.float32))

    metrics = {"model": "tf_mlp", "val_best_threshold": float(best_th), "val_at_best": best_row, "test": test_metrics}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Treinamento (TF-MLP) concluído.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Modelo salvo em:", os.path.abspath(keras_path))

if __name__ == "__main__":
    main()