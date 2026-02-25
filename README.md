# Sobre o Projeto
Visamos entender quais doenças podem estar atacando uma planta utilizando Redes Neurais Convolucionais (visão computacional) para analisar detalhes sobre as folhas das plantas e então identificar padrões de doenças.

## IAs - ResNet18 Pre Train
Com o objetivo de identificar padrões, todo o projeto de IA foi dividido em duas IAs com tarefas diferentes, a `Generalista` e a `Especialista`.
- Generalista: Analisa padrões físicos nas plantas, como bordas e formatos, visando identificar qual é o tipo de cultura que está sendo analisada (milho, trigo, soja). Esse tipo de operação diminui a sobrecarga de processamento em um só modelo IA, diminuindo erros e consequentemente, melhorando as predições.
    - (11.700.000) ~11,7M Parâmetros
- Especialista: Esse modelo é utilizado somente após a identificação da cultura específica, onde então a imagem é enviada a sua IA especialista, que irá trabalhar somente com informações sobre determinada cultura, isso melhora a predição e diminui o processamento.
    - (11.700.000) ~11,7M Parâmetros

## Objetivos Futuros
- Fazer melhorias na organização estrutural e código do projeto, adicionando classes e boas práticas a fim de melhorar a modularização e reusabilidade do código.
- Adicionar diferentes partes da planta, para uma possível análise completa.