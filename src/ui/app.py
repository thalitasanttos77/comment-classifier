import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import sys

HERE = os.path.abspath(os.path.dirname(__file__))     
SRC_DIR = os.path.abspath(os.path.join(HERE, ".."))   
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from embedder import embed_text, tokenize
    from model_io import load_resources, predict_proba_array
except Exception:
    from src.embedder import embed_text, tokenize  
    from src.model_io import load_resources, predict_proba_array  

# Heurísticas (opcional)
try:
    from heuristics import adjust_probability_with_heuristics
except Exception:
    adjust_probability_with_heuristics = None


# ---------------------------
# Utilidades
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_model_bundle(model_dir: str, strip_accents: bool):
    model, scaler, word_map, saved_th, model_type = load_resources(model_dir, strip_accents)
    vocab = set(word_map.keys())
    return {
        "model": model,
        "scaler": scaler,
        "word_map": word_map,
        "vocab": vocab,
        "saved_threshold": float(saved_th),
        "model_type": model_type,
        "model_dir": model_dir,
        "strip_accents": strip_accents,
    }

def compute_coverage(tokens, vocab):
    total = len(tokens)
    if total == 0:
        return 0.0
    known = sum(1 for t in tokens if t in vocab)
    return known / total

def classify_text(text: str, bundle, threshold: float, use_heuristics: bool, min_coverage: float | None):
    toks = tokenize(text, lowercase=True, strip_accents=bundle["strip_accents"])
    cov = compute_coverage(toks, bundle["vocab"])
    x = embed_text(text, bundle["word_map"], lowercase=True, strip_accents=bundle["strip_accents"]).reshape(1, -1)
    x_s = bundle["scaler"].transform(x)
    p_pos = float(predict_proba_array(bundle["model"], x_s, bundle["model_type"]).reshape(-1)[0])
    p_adj = p_pos
    if use_heuristics and adjust_probability_with_heuristics is not None:
        p_adj = float(adjust_probability_with_heuristics(p_adj, toks))
    forced_negative = False
    if min_coverage is not None and cov < float(min_coverage):
        p_adj = 0.0
        forced_negative = True
    pred = 1 if p_adj >= threshold else 0
    return {
        "prob_pos_raw": p_pos,
        "prob_pos_adj": p_adj,
        "coverage": cov,
        "pred": pred,
        "forced_negative": forced_negative,
        "tokens": toks,
    }

def list_model_dirs(base_dir="."):
    candidates = []
    names = ["models", "models_prec", "models_prec85"]
    for name in names:
        p = os.path.abspath(os.path.join(base_dir, name))
        if os.path.isdir(p):
            candidates.append(p)

    here = os.path.abspath(os.curdir)
    parent = os.path.dirname(here)
    try:
        for entry in os.listdir(parent):
            full = os.path.join(parent, entry)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "model.pkl")):
                if full not in candidates:
                    candidates.append(full)
    except Exception:
        pass
    return sorted(set(candidates))


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Classificador de Comentários", page_icon="💬", layout="wide")

st.title("💬Classificador de Comentários💬")

with st.sidebar:
    st.header("Configurações")
    project_root = os.path.abspath(os.path.join(HERE, "..", ".."))
    model_options = list_model_dirs(base_dir=project_root)

    default_dir = None
    for opt in model_options:
        if os.path.basename(opt) == "models":
            default_dir = opt
            break

    placeholder = "<selecione manualmente>"
    options_for_select = model_options if model_options else [placeholder]

    selected_model_dir = st.selectbox(
        "Diretório do modelo (model.pkl)",
        options=options_for_select,
        index=options_for_select.index(default_dir) if (default_dir and default_dir in options_for_select) else 0
    )

    strip_accents = st.checkbox("Remover acentos (strip_accents)", value=True,
                                help="Deve refletir como o vocabulário foi construído.")
    if not model_options or selected_model_dir == placeholder:
        st.warning(
            "Nenhum diretório de modelo encontrado automaticamente. "
            "Coloque uma pasta `models` (ou `models_prec`/`models_prec85`) na raiz do projeto "
            "com `model.pkl`, ou selecione manualmente o caminho."
        )
        st.caption(f"Procurando em: {project_root}")
        st.stop()
    with st.spinner("Carregando modelo..."):
        try:
            bundle = load_model_bundle(selected_model_dir, strip_accents)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo em {selected_model_dir}: {e}")
            st.exception(e)
            st.stop()

    st.markdown(f"• Modelo: `{os.path.basename(bundle['model_dir'])}`")
    st.markdown(f"• Threshold salvo: `{bundle['saved_threshold']:.3f}`")

    use_saved_threshold = st.checkbox("Usar threshold salvo no modelo", value=True)
    threshold = bundle["saved_threshold"]
    if not use_saved_threshold:
        threshold = st.slider("Threshold manual", min_value=0.30, max_value=0.95, value=float(threshold), step=0.01)

    use_heur = st.checkbox("Aplicar heurísticas conservadoras", value=True,
                           help="Ajusta probabilidade com regras simples (se disponível).")

    min_cov_on = st.checkbox("Aplicar cobertura mínima", value=False,
                             help="Força negativo quando a cobertura de vocabulário é baixa.")
    min_cov = st.slider("Cobertura mínima", 0.0, 1.0, 0.30, 0.01) if min_cov_on else None

    st.divider()
    st.caption("Dica: escolha `models` para o Modelo 0 (mais equilibrado). "
               "Você pode testar outros modelos como `models_prec` e `models_prec85`.")

tab1, tab2 = st.tabs(["🧍 Um a um", "📄 Arquivo (CSV)"])

# ---------------------------
# TAB 1: Um a um
# ---------------------------
with tab1:
    st.subheader("Classificar um comentário")
    text = st.text_area("Digite o comentário", height=140, placeholder="Cole ou digite um comentário aqui...")
    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        run_btn = st.button("Classificar")
    with col_b:
        clear_btn = st.button("Limpar")

    if clear_btn:
        st.experimental_rerun()

    if run_btn and text.strip():
        t0 = time.time()
        result = classify_text(
            text.strip(),
            bundle=bundle,
            threshold=threshold,
            use_heuristics=use_heur,
            min_coverage=min_cov
        )
        dt = (time.time() - t0)*1000

        pred_label = "positivo" if result["pred"] == 1 else "negativo"
        st.markdown(f"**Classificação:** {pred_label}")
        st.markdown(
            f"Probabilidade positiva (raw): `{result['prob_pos_raw']:.3f}`  | "
            f"Ajustada: `{result['prob_pos_adj']:.3f}`  | "
            f"Threshold: `{threshold:.3f}`  | "
            f"Cobertura: `{result['coverage']:.2f}`  | "
            f"Tempo: `{dt:.1f} ms`"
        )
        if result["forced_negative"]:
            st.info("Cobertura abaixo do mínimo → predição forçada para negativo.")

# ---------------------------
# TAB 2: CSV
# ---------------------------
with tab2:
    st.subheader("Classificar arquivo CSV")
    st.caption("O CSV deve ter a coluna 'texto'. Se tiver 'label' (0/1), as métricas serão calculadas.")
    upl = st.file_uploader("Envie um arquivo .csv", type=["csv"])

    if upl is not None:
        try:
            df_in = pd.read_csv(upl, encoding="utf-8")
        except UnicodeDecodeError:
            df_in = pd.read_csv(upl, encoding="latin-1")

        if "texto" not in df_in.columns:
            st.error("CSV precisa ter a coluna 'texto'.")
        else:
            st.write(f"Linhas lidas: {len(df_in)}")
            probs_raw = []
            probs_adj = []
            preds = []
            covs = []
            forced_list = []
            B = 512
            texts = df_in["texto"].fillna("").astype(str).tolist()
            for i in range(0, len(texts), B):
                batch = texts[i:i+B]
                toks_batch = [tokenize(t, lowercase=True, strip_accents=bundle["strip_accents"]) for t in batch]
                cov_batch = [compute_coverage(toks, bundle["vocab"]) for toks in toks_batch]
                X = np.stack([embed_text(t, bundle["word_map"], lowercase=True, strip_accents=bundle["strip_accents"]) for t in batch], axis=0)
                Xs = bundle["scaler"].transform(X)
                p_raw = predict_proba_array(bundle["model"], Xs, bundle["model_type"]).reshape(-1).astype(float)

                if use_heur and adjust_probability_with_heuristics is not None:
                    p_adj = np.array([adjust_probability_with_heuristics(float(p), toks) for p, toks in zip(p_raw, toks_batch)], dtype=float)
                else:
                    p_adj = p_raw.copy()

                if min_cov is not None:
                    mask_force = np.array([c < float(min_cov) for c in cov_batch], dtype=bool)
                    p_adj[mask_force] = 0.0
                else:
                    mask_force = np.zeros_like(p_adj, dtype=bool)

                y_hat = (p_adj >= threshold).astype(int)

                probs_raw.extend(p_raw.tolist())
                probs_adj.extend(p_adj.tolist())
                preds.extend(y_hat.tolist())
                covs.extend(cov_batch)
                forced_list.extend(mask_force.tolist())

            df_out = df_in.copy()
            df_out["prob_pos_raw"] = probs_raw
            df_out["prob_pos_adj"] = probs_adj
            df_out["pred"] = preds
            df_out["coverage"] = covs
            df_out["forced_negative"] = forced_list

            st.dataframe(df_out.head(20), use_container_width=True)
            if "label" in df_out.columns:
                y_true = df_out["label"].astype(int).values
                y_pred = df_out["pred"].astype(int).values
                TP = int(((y_true==1) & (y_pred==1)).sum())
                TN = int(((y_true==0) & (y_pred==0)).sum())
                FP = int(((y_true==0) & (y_pred==1)).sum())
                FN = int(((y_true==1) & (y_pred==0)).sum())
                precision = TP / (TP + FP) if (TP+FP)>0 else 0.0
                recall = TP / (TP + FN) if (TP+FN)>0 else 0.0
                f1 = 0.0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
                tnr = TN / (TN + FP) if (TN+FP)>0 else 0.0
                st.markdown("#### Métricas (seu CSV)")
                st.write({
                    "threshold": round(threshold, 3),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "specificity (TNR)": round(tnr, 4),
                    "confusion_matrix": [[TN, FP],[FN, TP]],
                })

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar resultados (CSV)",
                data=csv_bytes,
                file_name="resultado_classificacao.csv",
                mime="text/csv"
            )

            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                df_out.to_excel(writer, sheet_name="Resultados", index=False)
            st.download_button(
                label="Baixar resultados (Excel)",
                data=out_buf.getvalue(),
                file_name="resultado_classificacao.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.caption(f"Modelo: {os.path.basename(bundle['model_dir'])} | Threshold: {threshold:.3f} "
           f"| strip_accents: {bundle['strip_accents']} | Heurísticas: {use_heur} "
           f"| Cobertura mínima: {min_cov if min_cov is not None else 'desativada'}")