import os
import argparse
import pickle
import time
import numpy as np
from embedder import build_word_map, embed_text, tokenize

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError as e:
    raise SystemExit("openpyxl não está instalado. Instale com:\n  pip install openpyxl") from e

# Heurística opcional
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

def autosize_columns(ws):
    for col_idx, col in enumerate(ws.columns, start=1):
        max_len = 0
        for cell in col:
            val = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(val))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 60)

def safe_save_workbook(wb, path: str) -> str:
    try:
        wb.save(path)
        return os.path.abspath(path)
    except PermissionError:
        root, ext = os.path.splitext(path)
        ts = time.strftime("%Y%m%d-%H%M%S")
        alt = f"{root}_{ts}{ext or '.xlsx'}"
        wb.save(alt)
        return os.path.abspath(alt)

def main():
    parser = argparse.ArgumentParser(
        description="Classifica um arquivo (1 comentário por linha) e gera Excel (.xlsx): texto | classificacao | probabilidade."
    )
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--output_xlsx", type=str, default="predicoes.xlsx")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--sheet_name", type=str, default="Classificacoes")
    parser.add_argument("--threshold", type=float, default=None, help="Se não passar, usa o salvo no modelo")
    parser.add_argument("--use_heuristics", action="store_true", help="Aplicar regras simples (negação/léxico)")
    args = parser.parse_args()

    model, scaler, word_map, saved_th = load_resources(args.model_dir, args.strip_accents)
    decision_th = float(args.threshold) if args.threshold is not None else saved_th

    with open(args.input_txt, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    wb = Workbook()
    ws = wb.active
    ws.title = args.sheet_name

    headers = ["texto", "classificacao", "probabilidade"]
    ws.append(headers)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    center = Alignment(horizontal="center", vertical="center")
    for i in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=i)
        c.font = header_font
        c.fill = header_fill
        c.alignment = center

    has_proba = hasattr(model, "predict_proba")
    for text in texts:
        x = embed_text(text, word_map, lowercase=True, strip_accents=args.strip_accents).reshape(1, -1)
        x_s = scaler.transform(x)

        if has_proba:
            prob_pos = float(model.predict_proba(x_s)[0, 1])
            if args.use_heuristics and adjust_probability_with_heuristics is not None:
                toks = tokenize(text, lowercase=True, strip_accents=args.strip_accents)
                prob_pos = adjust_probability_with_heuristics(prob_pos, toks)
            pred = 1 if prob_pos >= decision_th else 0
            label = "positivo" if pred == 1 else "negativo"
            conf = prob_pos if pred == 1 else (1.0 - prob_pos)
            prob_cell_value = conf
        else:
            pred = int(model.predict(x_s)[0])
            label = "positivo" if pred == 1 else "negativo"
            prob_cell_value = None

        r = ws.max_row + 1
        ws.cell(row=r, column=1, value=text)
        ws.cell(row=r, column=2, value=label)
        c3 = ws.cell(row=r, column=3, value=prob_cell_value if prob_cell_value is not None else None)
        if prob_cell_value is not None:
            c3.number_format = "0%"
            c3.alignment = center
        else:
            c3.value = "N/D"
            c3.alignment = center

    autosize_columns(ws)

    out = args.output_xlsx
    if not out.lower().endswith(".xlsx"):
        out += ".xlsx"
    abs_path = safe_save_workbook(wb, out)
    print(f"Planilha Excel salva em: {abs_path}")

if __name__ == "__main__":
    main()