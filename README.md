# Sobre o Projeto
Visamos entender quais doenças podem estar atacando uma planta utilizando Redes Neurais Convolucionais (visão computacional) para analisar detalhes sobre as folhas das plantas e então identificar padrões de doenças.

## IAs
Com o objetivo de identificar padrões, todo o projeto de IA foi dividido em duas IAs com tarefas diferentes, a `Generalista` e a `Especialista`.
- Generalista: Analisa padrões físicos nas plantas, como bordas e formatos, visando identificar qual é o tipo de cultura que está sendo analisada (milho, trigo, soja). Esse tipo de operação diminui a sobrecarga de processamento em um só modelo IA, diminuindo erros e consequentemente, melhorando as predições.
    - (3.559.899) ~3.56M Parâmetros
- Especialista: Esse modelo é utilizado somente após a identificação da cultura específica, onde então a imagem é enviada a sua IA especialista, que irá trabalhar somente com informações sobre determinada cultura, isso melhora a predição e diminui o processamento.
    - (3.560.560) ~3.56M Parâmetros

## Objetivos Futuros
- Aprimorar a predição do projeto, otimizando os modelos e aprimorando o pré-processamento da imagem, identificando pontos ideais de trabalho (onde de fato a planta se encontra) ou aplicando diferentes filtros conforme análise prévia da imagem.
- Adicionar novas culturas e também a possibilidade de não identificação da cultura ou doênça.