import os
import pickle
import numpy as np

try:
    import tensorflow as tf  
    _TF_OK = True
except Exception:
    tf = None 
    _TF_OK = False

from embedder import build_word_map

def load_resources(model_dir: str, strip_accents: bool):
    """
    Carrega bundle (model.pkl), o vocabulário e retorna:
      - model: sklearn ou tf.keras.Model
      - scaler: StandardScaler salvo no pickle
      - word_map: dict token->vetor (para embedder)
      - threshold: float (decision_threshold)
      - model_type: str ("tf_keras" ou classe sklearn)
    """
    model_pkl = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_pkl):
        raise SystemExit(f"Não encontrado: {model_pkl}")

    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)

    words = np.load(os.path.join(model_dir, "words.npy"), allow_pickle=True).tolist()
    word_vectors = np.load(os.path.join(model_dir, "word_vectors.npy"))
    word_map = build_word_map(words, word_vectors, lowercase=True, strip_accents=strip_accents)

    model_type = bundle.get("model_type", None)
    threshold = float(bundle.get("decision_threshold", 0.5))
    scaler = bundle.get("scaler", None)

    if model_type == "tf_keras":
        if not _TF_OK:
            raise SystemExit("TensorFlow não instalado. Instale 'tensorflow-cpu' ou 'tensorflow'.")
        model_path = bundle.get("model_path", "tf_model.keras")
        abs_path = os.path.join(model_dir, model_path)
        if not os.path.exists(abs_path):
            raise SystemExit(f"Modelo Keras não encontrado: {abs_path}")
        model = tf.keras.models.load_model(abs_path)
    else:
        model = bundle.get("model")
        model_type = model_type or type(model).__name__

    return model, scaler, word_map, threshold, model_type

def predict_proba_array(model, X_s: np.ndarray, model_type: str) -> np.ndarray:
    """
    Retorna probabilidades de classe positiva (shape (n,)).
    Suporta: tf_keras (saída sigmoid), sklearn com predict_proba ou decision_function.
    """
    if model_type == "tf_keras":
        proba = model.predict(X_s, verbose=0).reshape(-1).astype(float)
        return proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_s)[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X_s).astype(float)
        return 1.0 / (1.0 + np.exp(-z))  
    pred = np.asarray(model.predict(X_s)).reshape(-1)
    return pred.astype(float)