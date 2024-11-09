# 🏥 Diagnóstico e Otimização de Processos Hospitalares

## Contexto

Este projeto desenvolve uma ferramenta analítica completa para diagnóstico e otimização de processos hospitalares, utilizando Ciência de Dados, Teoria das Filas, Teoria das Restrições (TOC) e princípios Lean Healthcare. 

🔗 [Clique e acesse o desenvolvimento da aplicação](https://leanhospitais.streamlit.app/)

## 📺 Demonstração

🎥 [Assista ao vídeo de demonstração no YouTube](https://youtu.be/bVbeE5P8gYk)


### Estrutura de Dados
A aplicação funciona a partir do upload de um arquivo Excel padronizado (diagnostico_hospitalar.xlsx - [**Arquivo**][file-link]) contendo dados estruturados sobre diferentes setores e processos hospitalares. Por questões de confidencialidade e direitos autorais, uma versão demonstrativa do arquivo pode ser solicitada diretamente ao autor.

[file-link]: https://docs.google.com/spreadsheets/d/1kaAEsZGXAhvVh5NXUxamSsnPLn6_6l5V/edit?usp=sharing&ouid=100127589008142410245&rtpof=true&sd=true "Arquivo Exemplo - Acesso Restrito"

O arquivo deve conter informações específicas distribuídas em diferentes abas:
* Dados de fluxo de pacientes (mensais, semanais e por hora)
* Informações de triagem e classificação de risco
* Métricas do centro cirúrgico
* Dados de consultas médicas
* Informações de internação e ocupação
* Métricas de SADT (Serviço de Apoio Diagnóstico e Terapêutico)

### Processamento e Análise
A ferramenta realiza análises automáticas para:
* Identificar gargalos operacionais
* Otimizar alocação de recursos
* Melhorar o fluxo de pacientes
* Gerar visualizações interativas
* Calcular métricas-chave de performance

A estrutura de dados padronizada permite uma análise consistente e comparável entre diferentes períodos ou unidades hospitalares, facilitando a identificação de oportunidades de melhoria e a tomada de decisão baseada em dados.

## Resultados e Benefícios Alcançados
* **Redução de Tempo de Espera:**
  * Otimização do tempo porta-médico
  * Diminuição do tempo de espera na triagem
  * Melhor gestão de filas por especialidade

* **Otimização de Recursos:**
  * Alocação eficiente de médicos por especialidade
  * Melhor utilização de salas cirúrgicas
  * Distribuição otimizada de leitos

* **Melhorias Operacionais:**
  * Redução nas taxas de cancelamento cirúrgico
  * Diminuição do tempo de setup de salas
  * Melhor gestão da capacidade instalada

* **Gestão Baseada em Dados:**
  * Dashboards interativos para tomada de decisão
  * Previsibilidade de demanda
  * Métricas claras de desempenho

## Funcionalidades Principais

### 1. Análise da Porta de Entrada
* Monitoramento de fluxo de pacientes (hora/dia/mês)
* Análise de pacientes verticais vs. horizontais
* Distribuição por classificação de risco
* Gestão de pontos de cuidado

### 2. Gestão de Triagem
* Análise de distribuição por urgência
* Monitoramento de enfermeiros por horário
* Gestão de salas de triagem
* Tempos médios de atendimento

### 3. Gestão de Consultas
* Análise de tempo por etapa da consulta
* Distribuição de médicos por horário
* Análise de demanda por especialidade
* Gestão de capacidade médica

### 4. Gestão do Centro Cirúrgico
* Análise de eficiência operacional
* Monitoramento de cancelamentos
* Gestão de salas cirúrgicas
* Análise de tempos cirúrgicos

### 5. Gestão de Internação
* Análise de ocupação por setor
* Gestão de demanda de leitos
* Monitoramento de tempo de permanência
* Análise de taxas de internação

## Tecnologias e Bibliotecas
* **Framework Principal:** Python 3.7+
* **Interface:** Streamlit
* **Visualização:** 
  * Plotly
  * Graphviz
  * Matplotlib
* **Análise de Dados:** 
  * Pandas
  * NumPy
  * Statsmodels
* **Controle de Versão:** Git/GitHub

## Modelos e Métodos Analíticos
* **Teoria das Filas:**
  * Modelos M/M/c
  * Análise de capacidade
  * Cálculo de tempos de espera

* **Lean Healthcare:**
  * VSM (Value Stream Mapping)
  * Análise de gargalos
  * Identificação de desperdícios

* **Análise Estatística:**
  * Previsão de demanda
  * Análise de tendências
  * Correlação entre variáveis

## Próximos Passos
1. **Análise Preditiva:**
   * Previsão de demanda por especialidade
   * Modelo de previsão de internações
   * Otimização automática de escalas

2. **Integração de Sistemas:**
   * Conexão com sistemas hospitalares
   * API para dados em tempo real
   * Automação de relatórios

3. **Novas Funcionalidades:**
   * Módulo de gestão de equipamentos
   * Análise de custos operacionais
   * Dashboard mobile

## Autor e Direitos de Uso

**Autor:** Francisco Pena

**Direitos Autorais:** © 2024 Francisco Pena. Todos os direitos reservados.

Este projeto e todo seu conteúdo, incluindo mas não limitado a código fonte, documentação, visualizações e metodologias, são de autoria de Francisco Pena. Qualquer reprodução, distribuição, modificação ou uso deste material, no todo ou em parte, requer autorização prévia e expressa do autor.

### Contato
Para solicitar permissão de uso ou discutir possíveis colaborações:
- LinkedIn: [Francisco Pena](https://www.linkedin.com/in/franciscobahiapena/)
- Email: bahiapenafrancisco@gmail.com

**Nota:** Este projeto foi desenvolvido como parte de um portfólio profissional e sua utilização está sujeita a termos específicos. Para uso comercial ou acadêmico, consulte o autor.
