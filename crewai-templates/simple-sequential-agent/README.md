# Explicação do Template
Esse é um template básico de um agente e uma tarefa apenas, mas que pode ser escalado com a inserção de mais agentes, nesse caso utilizando uma arquitetura sequencial. Os parâmetros relevantes estão expostos e podem ser ajustados para melhor comportamento do agente.

## Como executar o template?
Para executar o template você pode utilizar o seguinte comando:

```uv run main.py```

Esse comando irá instalar as bibliotecas e executar o arquivo `main.py`, com as dependências definidas no arquivo `pyproject.toml`.

## Como modificar o template?
Para modificar o template, tente entender como o setup do agente funciona, e a partir disso você pode experimentar modificar alguns dos seus prompts, e também da sua task, de modo a entender melhor o framework. Você pode também adicionar mais tasks e agentes, pra entender como isso funciona e também brincar com parâmetros de definição do agente, como o `allow_delegation`, também modificar a temperatura do LLM (`temperature`), e por aí vai.

---
**Se você precisar de qualquer ajuda, ou notar algum problema com o template, por favor fique à vontade para abrir uma pull request com a solução e o detalhamente, ou se você não conseguir consertá-lo, fique à vontade para abrir uma issue com o detalhamento do problema. 



