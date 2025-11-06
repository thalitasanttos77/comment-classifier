# CommentClassifier: Classificador de Comentários

📜 Descrição do Projeto  
Este projeto implementa um classificador de comentários (positivo/negativo) desenvolvido como avaliação da disciplina Inteligência Artificial I (DS803) — Tecnologia em Análise e Desenvolvimento de Sistemas (UFPR), orientado pelo Prof. Dr. Roberto Tadeu Raittz. O sistema treina modelos a partir de vetores textuais e disponibiliza ferramentas para inferência individual, processamento em lote, avaliação e treino com diferentes objetivos (ex.: aumentar precisão).

---

## 🎯 Objetivos
- Treinar e testar um modelo de classificação com os dados fornecidos.
- Validar o modelo com novos textos rotulados para medir performance.
- Fornecer ferramentas (scripts + UI) para classificar comentários individualmente e em lote.
- Permitir re-treino com estratégias para aumentar precisão (ajuste de threshold, bias de classe).

---

## 🗂️ Conjunto de Dados (Dataset)
Os dados usados no projeto estão na pasta `DADOS` e contêm:
- `PALAVRASpc.txt`: Lista de palavras vetorizadas.
- `WWRDpc.dat`: Vetores (100 dimensões) para vocabulário.
- `WTEXpc.dat`: Vetores (100 dimensões) para textos (média dos vetores das palavras).
- `CLtx.dat`: Rótulos dos textos de `WTEXpc.dat` (1 = positivo, 0 = negativo).

---

## 📦 Artefatos do Modelo (o que o diretório `models` deve conter)
Um diretório de modelo típico (ex.: `models`, `models_prec`, `models_prec85`) deve conter pelo menos:
- `model.pkl` — o classificador treinado
- `scaler.pkl` — scaler usado para normalizar embeddings
- `word_map.json` (ou equivalente) — mapeamento token → índice/vetor
- Arquivo com threshold salvo (ou metadados) dependendo da implementação

O app procura por pastas com nome `models`, `models_prec` ou `models_prec85` na raiz do projeto.

---

## Instalação
Recomenda-se criar um ambiente virtual e instalar dependências:

Windows (PowerShell)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Não faça commit do diretório do virtualenv (`venv`, `.venv`, etc.) no repositório — adicione-o ao `.gitignore`.

Exemplo mínimo `.gitignore` (adicione na raiz):
```
venv/
.venv/
__pycache__/
*.py[cod]
.vscode/
.idea/
.DS_Store
```

---

## Uso — comandos rápidos (exemplos fornecidos)

Observação: muitos exemplos usam caminhos Windows com `..\`. Ajuste para caminhos Unix (`../`) se necessário.

- Instalar dependências:
```
pip install -r requirements.txt
```

Inferência (um a um)
```
python infer.py --model_dir ..\models --strip_accents --use_heuristics
python infer.py --model_dir ..\models_prec --strip_accents --use_heuristics
python infer.py --model_dir ..\models_prec85 --strip_accents --use_heuristics
```

Classificação em lote (arquivo de texto com um comentário por linha)
```
python batch_classify.py --model_dir ..\models --input_txt ..\lote-comentarios-positivos.txt --output_csv ..\resultado_lote.csv --strip_accents

python batch_classify.py --model_dir ..\models --input_txt ..\lote-comentarios-positivos.txt --output_xlsx ..\resultado_lote.xlsx --strip_accents

python batch_classify.py --model_dir ..\models --input_txt ..\comentarios-mistos.txt --output_xlsx ..\resultado_lote.xlsx --strip_accents
```

Classificação em lote com heurística e/ou modelo focado em precisão
```
python batch_classify.py --model_dir ..\models --input_txt ..\comentarios-mistos.txt --output_xlsx ..\resultado.xlsx --strip_accents --use_heuristics

python batch_classify.py --model_dir ..\models_prec85 --input_txt ..\comentarios-mistos.txt --output_xlsx ..\resultado.xlsx --strip_accents --use_heuristics
```

Treino
```
# Treinar (padrão)
python train.py --data_dir ..\dados --out_dir ..\models --calibrate sigmoid

# Treinar na pasta src:
python src/train.py --data_dir ..\dados --out_dir ..\models_tf
```

Retreinar com foco em precisão / controle de recall e bias negativo
```
# Ajustar para aumentar precisão com restrição de recall ≥ 0.70
python train.py --data_dir ..\dados --out_dir ..\models_prec --metric precision_at_recall --recall_min 0.70 --neg_bias 1.8

# Modelo ainda mais focado em precisão (ex.: 85% de precisão alvo)
python train.py --data_dir ..\dados --out_dir ..\models_prec85 --metric precision --neg_bias 3.0
```

Avaliação (CSV rotulado)
```
# Com heurísticas
python evaluate_labeled.py --model_dir ..\models --input_csv ..\comentarios_classificados.csv --strip_accents --use_heuristics

# Sem heurísticas
python evaluate_labeled.py --model_dir ..\models --input_csv ..\comentarios_classificados.csv --strip_accents
```

Avaliar modelos e salvar saída (ex.: gerar JSON)
```
# Modelo 0
python evaluate_labeled.py --model_dir ..\models --input_csv ..\comentarios_classificados.csv --strip_accents > ..\eval_models_prec.json

# Modelo 1
python evaluate_labeled.py --model_dir ..\models_prec --input_csv ..\comentarios_classificados.csv --strip_accents > ..\eval_models_prec.json

# Modelo 2
python evaluate_labeled.py --model_dir ..\models_prec85 --input_csv ..\comentarios_classificados.csv --strip_accents > ..\eval_models_prec85.json
```

Executar a interface Streamlit
```
# Rodar a partir da pasta src (cd src && ...)
streamlit run ui/app.py

# Rodar a partir da raiz do projeto
streamlit run src\ui\app.py
```

---

## O que cada flag faz (resumo)
- `--data_dir ..\dados` : usa os arquivos WTEXpc, CLtx, etc. dessa pasta para treinar.  
- `--out_dir ..\models_prec` : salva o modelo e artefatos (scaler, threshold, vocabulário) nessa pasta.  
- `--metric precision_at_recall` : escolhe o threshold na validação maximizando precisão, sujeita à restrição de recall.  
- `--recall_min 0.70` : ao escolher o threshold, só considera candidatos cuja sensibilidade (recall) ≥ 0.70.  
- `--neg_bias 1.8` : aumenta o peso da classe negativa no treino (reduz falsos positivos; tende a elevar precisão).  
- `--strip_accents` : remover acentos no pré-processamento (deve refletir como o vocabulário foi construído).  
- `--use_heuristics` : aplica ajustes de probabilidade via heurísticas (regras) antes de decidir o rótulo final.  
- `--calibrate sigmoid` : aplicar calibração de probabilidades (ex.: Platt / Sigmoid) durante o treino.

---

## Observações sobre heurísticas
- Heurísticas costumam melhorar recall/precision em casos onde o modelo estatístico erra sistematicamente (palavras-chave, baixa cobertura do vocabulário, negações).  
- Se o desempenho cair muito sem heurísticas, convém:
  - rodar avaliação comparativa (com/sem heurística) usando `evaluate_labeled.py` e analisar os casos que mudam;
  - usar heurística somente quando o modelo estiver incerto (ex.: p_raw em [0.35, 0.65]);
  - transformar regras em features e treinar um meta-classificador (stacking) para reduzir viés manual.
- O app já calcula `coverage` (fração de tokens conhecidos); se baixa, heurísticas podem ser críticas.

---

## Exemplos de comentários (para testes)
Positivos:
- "Absolutamente incrível! O seu trabalho realmente se destaca pela qualidade e pelo cuidado nos detalhes. Fiquei muito tempo analisando e é impressionante. Continue assim, está fantástico!"
- "Você faz parecer fácil! A complexidade do código foi tratada com uma maestria incrível. Dá para ver que você realmente domina o assunto. Talentoso demais."

Negativos:
- "Infelizmente, o trabalho parece ter sido feito às pressas. Notei vários erros de edição que comprometem o resultado final."
- "O som está horrível. Há muito ruído de fundo e a música está mais alta que a voz. Tive que desistir de assistir"

---

## Recomendações de workflow
1. Criar ambiente virtual e instalar dependências (veja seção Instalação).  
2. Preparar a pasta `models/` com os artefatos do modelo.  
3. Rodar `streamlit run src\ui\app.py` ou usar os scripts de inferência local (`infer.py`, `batch_classify.py`) para testar.  
4. Avaliar em CSV rotulado com `evaluate_labeled.py` para medir métricas com/sem heurísticas.  
5. Retreinar modelos com `train.py` ajustando `--metric` e `--neg_bias` conforme necessário para atingir a precisão desejada.
