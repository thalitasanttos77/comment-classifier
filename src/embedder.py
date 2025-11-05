import re
import unicodedata
from typing import Dict, Iterable, List
import numpy as np

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def remove_accents(text: str) -> str:
    # Usa só biblioteca padrão
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

def tokenize(text: str, lowercase: bool = True, strip_accents: bool = False) -> List[str]:
    if lowercase:
        text = text.lower()
    if strip_accents:
        text = remove_accents(text)
    return _WORD_RE.findall(text)

def build_word_map(words: list[str], word_vectors: np.ndarray, lowercase: bool = True, strip_accents: bool = False) -> Dict[str, np.ndarray]:
    mapping: Dict[str, np.ndarray] = {}
    for w, vec in zip(words, word_vectors):
        token = w.lower() if lowercase else w
        token = remove_accents(token) if strip_accents else token
        mapping[token] = vec.astype(np.float32)
    return mapping

def embed_text(text: str, word_map: Dict[str, np.ndarray], lowercase: bool = True, strip_accents: bool = False) -> np.ndarray:
    tokens = tokenize(text, lowercase=lowercase, strip_accents=strip_accents)
    vecs = [word_map[t] for t in tokens if t in word_map]
    if not vecs:
        # Sem palavras conhecidas → vetor zero (100D)
        dim = next(iter(word_map.values())).shape[0]
        return np.zeros((dim,), dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

def embed_many(texts: Iterable[str], word_map: Dict[str, np.ndarray], lowercase: bool = True, strip_accents: bool = False) -> np.ndarray:
    return np.stack([embed_text(t, word_map, lowercase, strip_accents) for t in texts], axis=0)