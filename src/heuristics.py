from typing import List

POS_WORDS = {
     "bom", "boa", "excelente", "otimo", "ótimo", "gostei", "adorei",
    "maravilhoso", "maravilhosa", "perfeito", "recomendo", "lindo", "linda",
    "incrivel", "incrível", "fantastico", "fantástico", "show", "sensacional", "top",
    "coragem", "corajoso", "corajosa", "honesto", "honesta", "integro", "íntegro", "integra", "integridade",
    "justo", "justa", "etico", "ético", "respeitoso", "humilde", "humildade", "empatico", "empática",
    "comprometido", "comprometida", "admiravel", "admirável", "exemplar", "de palavra", "responsavel", "responsável", "bem", "legal", "agradavel", "agradável", "bem recebido", "bem recebida", "aceito", "aceita" , "sucesso"
}
NEG_WORDS = {
    "ruim", "pessimo", "péssimo", "horrivel", "horrível", "odiei",
    "lixo", "terrivel", "terrível", "detestei", "horroroso", "bosta",
    "merda", "péssima", "pessima", "decepcionante", "nojento", "nojenta"
}
NEGATORS = {"nao", "não", "nem", "nunca", "jamais"}

def adjust_probability_with_heuristics(prob_pos: float, tokens: List[str]) -> float:
    """
    Ajusta levemente a probabilidade de positivo usando regras simples:
    - Negador + palavra positiva: derruba probabilidade de positivo (ex.: "não gostei").
    - Presença de palavras negativas fortes: puxa para negativo.
    - Somente palavras positivas (sem negador): puxa um pouco para positivo.
    Observação: isso é um pós-processamento na inferência; não altera o treinamento.
    """
    toks = [t.lower() for t in tokens]
    has_negator = any(t in NEGATORS for t in toks)
    has_pos = any(t in POS_WORDS for t in toks)
    has_neg = any(t in NEG_WORDS for t in toks)

    p = prob_pos
    if has_negator and has_pos and not has_neg:
        p = min(p, 0.25)   
    if has_neg and not has_pos:
        p = min(p, 0.35)    
    if has_pos and not has_neg and not has_negator:
        p = max(p, 0.65)   
    return max(0.0, min(1.0, p))