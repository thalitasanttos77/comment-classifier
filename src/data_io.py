import os
import numpy as np
from typing import List, Tuple

def load_words(words_path: str) -> List[str]:
    with open(words_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_matrix(dat_path: str, dtype=np.float32) -> np.ndarray:
    # Arquivos .dat com números separados por espaço
    return np.loadtxt(dat_path, dtype=dtype)

def load_labels(labels_path: str, dtype=int) -> np.ndarray:
    return np.loadtxt(labels_path, dtype=dtype)

def load_all(data_dir: str) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Espera em data_dir:
      - PALAVRASpc.txt
      - WWRDpc.dat     (n_palavras x 100)
      - WTEXpc.dat     (n_textos x 100)
      - CLtx.dat       (n_textos) com 0/1
    """
    words_path = os.path.join(data_dir, "PALAVRASpc.txt")
    wwr_path   = os.path.join(data_dir, "WWRDpc.dat")
    wtex_path  = os.path.join(data_dir, "WTEXpc.dat")
    cltx_path  = os.path.join(data_dir, "CLtx.dat")

    if not all(os.path.exists(p) for p in [words_path, wwr_path, wtex_path, cltx_path]):
        raise SystemExit("Arquivos não encontrados em data_dir. Esperado: PALAVRASpc.txt, WWRDpc.dat, WTEXpc.dat, CLtx.dat")

    words = load_words(words_path)
    word_vectors = load_matrix(wwr_path)
    text_vectors = load_matrix(wtex_path)
    labels = load_labels(cltx_path)

    assert word_vectors.ndim == 2 and word_vectors.shape[1] == 100, "Vetores de palavras devem ter 100 dimensões."
    assert text_vectors.ndim == 2 and text_vectors.shape[1] == 100, "Vetores de textos devem ter 100 dimensões."
    assert len(words) == word_vectors.shape[0], "Número de palavras e vetores não batem."
    assert text_vectors.shape[0] == labels.shape[0], "Textos e rótulos não batem."

    return words, word_vectors, text_vectors, labels