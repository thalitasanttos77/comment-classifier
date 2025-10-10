# CommentClassifier: Classificador de Comentários

## 📜 Descrição do Projeto
[cite_start]Este projeto tem como proposta desenvolver um modelo de reconhecimento de padrões para classificar comentários[cite: 8]. Utilizando uma base de dados pré-processada, o modelo aprende a distinguir textos com conotação positiva e negativa.

[cite_start]Este trabalho foi desenvolvido como uma das avaliações na disciplina de **Inteligência Computacional Aplicada I (DS803)** do curso de Tecnologia em Análise e Desenvolvimento de Sistemas da UFPR, sob orientação do Prof. Dr. Roberto Tadeu Raittz[cite: 28].

## 🎯 Objetivos
O roteiro de desenvolvimento do projeto consiste nas seguintes etapas:
* [cite_start]**Treinar e testar** um modelo de classificação com os dados fornecidos[cite: 10].
* [cite_start]**Validar** o modelo com um novo conjunto de textos (positivos e negativos) para verificar sua performance em casos reais[cite: 11].
* [cite_start]**Discutir** os resultados obtidos[cite: 12].
* [cite_start]**Desenvolver** uma ferramenta de classificação executável que permita a entrada de texto livre pelo usuário para classificação individual[cite: 13].

## 🗂️ Conjunto de Dados (Dataset)
[cite_start]Os dados utilizados para o treinamento e teste do modelo estão localizados na pasta `DADOS` e são compostos pelos seguintes arquivos[cite: 2]:

* [cite_start]`PALAVRASpc.txt`: Lista contendo 9.538 palavras vetorizadas[cite: 3].
* [cite_start]`WWRDpc.dat`: Vetores de 100 coordenadas correspondentes a cada palavra da lista anterior[cite: 4].
* `WTEXpc.dat`: Vetores de 100 coordenadas para 10.400 textos. [cite_start]Cada vetor representa a média dos vetores das palavras que o compõem[cite: 5].
* [cite_start]`CLtx.dat`: Classificação dos textos de `WTEXpc.dat`, onde `1` representa um texto positivo e `0` um texto negativo[cite: 6].

## 🛠️ Metodologia e Etapas do Relatório
[cite_start]A avaliação do projeto é baseada em um relatório impresso contendo as etapas do desenvolvimento[cite: 16, 18]:
1.  [cite_start]**Resumo** [cite: 19]
2.  [cite_start]**Apresentação e Introdução** [cite: 20]
3.  [cite_start]**Obtenção e Classificação dos Padrões** [cite: 21]
4.  [cite_start]**Extração de Características** [cite: 22]
5.  [cite_start]**Escolha do Classificador** [cite: 23]
6.  [cite_start]**Testes de Performance** [cite: 24]
7.  [cite_start]**Aplicação do Modelo** em comentários não utilizados no treinamento [cite: 25]
8.  [cite_start]**Conclusão** [cite: 26]

## 👨‍💻 Autores
[cite_start]O trabalho pode ser realizado em equipes de até três pessoas[cite: 14].

* [Seu Nome Completo]
* [Nome do Membro 2]
* [Nome do Membro 3]

---
