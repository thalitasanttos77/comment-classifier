# CommentClassifier: Classificador de Comentários

## 📜 Descrição do Projeto
Este projeto tem como proposta desenvolver um modelo de reconhecimento de padrões para classificar comentários. Utilizando uma base de dados pré-processada, o modelo aprende a distinguir textos com conotação positiva e negativa.

Este trabalho foi desenvolvido como uma das avaliações na disciplina de **Inteligência Artificial I (DS803)** do curso de Tecnologia em Análise e Desenvolvimento de Sistemas da UFPR, sob orientação do Prof. Dr. Roberto Tadeu Raittz.

## 🎯 Objetivos
O roteiro de desenvolvimento do projeto consiste nas seguintes etapas:
**Treinar e testar** um modelo de classificação com os dados fornecidos.
**Validar** o modelo com um novo conjunto de textos (positivos e negativos) para verificar sua performance em casos reais.
**Discutir** os resultados obtidos.
**Desenvolver** uma ferramenta de classificação executável que permita a entrada de texto livre pelo usuário para classificação individual.

## 🗂️ Conjunto de Dados (Dataset)
Os dados utilizados para o treinamento e teste do modelo estão localizados na pasta `DADOS` e são compostos pelos seguintes arquivos:

* `PALAVRASpc.txt`: Lista contendo 9.538 palavras vetorizadas.
* `WWRDpc.dat`: Vetores de 100 coordenadas correspondentes a cada palavra da lista anterior.
* `WTEXpc.dat`: Vetores de 100 coordenadas para 10.400 textos. Cada vetor representa a média dos vetores das palavras que o compõem.
* `CLtx.dat`: Classificação dos textos de `WTEXpc.dat`, onde `1` representa um texto positivo e `0` um texto negativo.

## 🛠️ Metodologia e Etapas do Relatório
A avaliação do projeto é baseada em um relatório impresso contendo as etapas do desenvolvimento:
1.  **Resumo**
2.  **Apresentação e Introdução**
3.  **Obtenção e Classificação dos Padrões** 
4.  **Extração de Características** 
5.  **Escolha do Classificador** 
6.  **Testes de Performance** 
7.  **Aplicação do Modelo** em comentários não utilizados no treinamento 
8.  **Conclusão** 
