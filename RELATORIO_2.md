# Relatório de Implementação — Gerador de Personas Sintéticas (HDT 7S)

## 1. Visão Geral

A aplicação gera dados sintéticos de pesquisa sobre **proteção tecnológica organizacional** utilizando o conceito de **Human Digital Twins (HDTs)** combinado com o **modelo 7S de McKinsey**. O objetivo é produzir respostas realistas a um questionário Likert (1–7) a partir de personas sintéticas cujos perfis demográficos refletem a distribuição de um dataset real.

A interface é construída com **Gradio** e executada via Jupyter Notebook (`00_ui_gerar_dados_sinteticos.ipynb`).

---

## 2. Arquitetura

```
00_ui_gerar_dados_sinteticos.ipynb   ← Ponto de entrada (lança a UI)
utils/
├── UIUtils.py                       ← Interface Gradio (apresentação e orquestração)
└── GerarDadosUtils.py               ← Lógica de negócio (geração de personas e respostas via LLM)
inputs/
└── questionario_real.csv            ← Dataset real de referência
outputs/                             ← CSVs gerados (personas, respostas, comparativos)
```

---

## 3. Classe `GerarDadosUtils` — Lógica de Geração

### 3.1 Configuração do LLM

O construtor aceita dois provedores: **Ollama** (local) ou **OpenAI** (API). O método `get_response` despacha o prompt para o provedor selecionado e retorna a resposta em texto (esperando JSON).

- `_get_response_ollama`: usa a biblioteca `ollama.Client` com parâmetros de seed, temperatura e `format="json"`.
- `_get_response_openai`: usa a biblioteca `openai.OpenAI` com temperatura configurável.

### 3.2 Formatação do Dataset Real

O método `formatar_dataset_real` lê o CSV de respostas reais, detecta automaticamente o formato (padrão ou legado com separador `","`), converte colunas numéricas, limpa strings categóricas e remove linhas vazias. Retorna um DataFrame padronizado com 32 itens Likert + 7 atributos demográficos.

### 3.3 Geração de Personas

O método `gerar_personas_df` cria N personas com atributos demográficos amostrados por **proporções ponderadas** (weighted random choices):

| Atributo | Método de amostragem |
|----------|---------------------|
| Idade | `random.randint(min, max)` |
| Sexo | `random.choices` com pesos |
| Setor de trabalho | `random.choices` com pesos |
| Porte organizacional | `random.choices` com pesos |
| Cargo/função | `random.choices` com pesos |
| Tempo de atuação | `random.randint(min, max)` |
| Treinamento recebido | `random.choices` com pesos |

As proporções padrão são derivadas do dataset real e podem ser sobrescritas pela UI.

### 3.4 Cálculo da Distribuição-Alvo

O método `calcular_target_probs` calcula, para cada item do questionário, a frequência relativa de cada valor Likert (1–7) no dataset real. Essa distribuição é incluída no prompt para guiar o LLM a gerar respostas com distribuição estatística semelhante à real.

### 3.5 Montagem do Prompt HDT

O método `montar_prompt_hdt` constrói um prompt detalhado contendo:

1. Instrução de role-play como Human Digital Twin brasileiro.
2. Perfil demográfico completo da persona (JSON).
3. As 32 afirmações do questionário 7S com dimensão e código.
4. Escala Likert explicada.
5. Regras de variabilidade e formato de saída (JSON estrito).
6. Distribuição-alvo por item para calibração estatística.

### 3.6 Geração de Respostas

O método `responder_persona`:

1. Monta o prompt com perfil + distribuição-alvo.
2. Calcula uma seed determinística por persona/sessão.
3. Envia ao LLM com `temperature=0.8`.
4. Extrai o JSON da resposta (com fallback via regex).
5. Normaliza as respostas em um dicionário plano com todos os itens.

### 3.7 Pipeline Completo

O método `executar` orquestra todo o fluxo em modo batch (sem UI):
formatar dados reais → gerar personas → calcular distribuição-alvo → gerar respostas (loop sessões × personas) → salvar CSVs.

---

## 4. Classe `UIUtils` — Interface Gradio

### 4.1 Estrutura de Abas

A interface possui 3 abas principais:

| Aba | Função |
|-----|--------|
| ℹ️ Sobre | Exibe documentação do projeto (Sobre.md) |
| 📂 Passo 1 - Dataset Real | Upload/inserção manual do CSV de respostas reais |
| 🧬 Passo 2 - Gerar Personas | Configuração de parâmetros e execução da geração |

### 4.2 Passo 1 — Gerenciamento do Dataset Real

- **Upload CSV**: o usuário envia um arquivo `.csv` que é validado e salvo em `inputs/questionario_real.csv`.
- **Inserção manual**: permite adicionar linhas individuais com 32 valores Likert + atributos demográficos via formulário.
- **Visualização**: exibe o dataset atual em tabela interativa.
- **Carregar proporções reais**: botão que lê o dataset e preenche automaticamente os campos de proporção na aba de geração.

### 4.3 Passo 2 — Geração de Personas Sintéticas

Parâmetros configuráveis:

- Seed, número de personas, sessões por persona.
- Provedor LLM (Ollama ou OpenAI) com host/modelo/API key.
- Faixas de idade e tempo de atuação.
- Opções e proporções para cada atributo categórico (editáveis via texto).
- Bins para agrupamento de idade e tempo nas estatísticas.

### 4.4 Execução Incremental (Generator)

O método `_executar_geracao` é um **Python generator** que faz `yield` a cada persona processada, permitindo atualização em tempo real na UI:

1. Formata o dataset real e salva versão padronizada.
2. Gera o DataFrame de personas.
3. Calcula estatísticas demográficas.
4. Itera sobre sessões × personas, chamando `responder_persona` para cada uma.
5. A cada iteração, atualiza: barra de progresso, tabela parcial de resultados, cards HTML com justificativas.
6. Ao final, salva todos os CSVs e gera gráficos comparativos.

### 4.5 Visualização de Resultados

- **Cards HTML**: cada persona é exibida com ícone de gênero, ID, idade, cargo, justificativa e tabela expansível de respostas com cores interpoladas (vermelho→verde conforme valor Likert).
- **Estatísticas demográficas**: tabela com contagem e percentual por atributo/categoria.
- **Gráficos comparativos** (Real vs. Sintético):
  - Distribuição de frequências Likert por dimensão (barras agrupadas).
  - Média por dimensão (barras agrupadas com linha neutra em 4).
- **PLS-SEM**: modelo de equações estruturais simplificado (mínimos quadrados) com bootstrap para p-values, exibido como diagrama de caminhos com coeficientes β e R².

### 4.6 Download

Ao final da geração, um arquivo CSV consolidado (`hdt_personas_respostas.csv`) fica disponível para download direto na interface.

---

## 5. Modelo Teórico — 7S de Proteção Tecnológica

O questionário contém 32 itens distribuídos em 8 dimensões:

| Dimensão | Itens | Tipo |
|----------|-------|------|
| Shared Values | SV1–SV4 | Independente |
| Strategy | SG1–SG4 | Independente |
| Structure | SU1–SU4 | Independente |
| Systems | SM1–SM4 | Independente |
| Staff | SF1–SF4 | Independente |
| Style | SY1–SY4 | Independente |
| Skills | SK1–SK4 | Independente |
| Technology Protection | TP1–TP4 | Dependente |

As 7 dimensões independentes (7S) são preditoras da variável dependente (Technology Protection). O modelo PLS-SEM estima os coeficientes de caminho (β) de cada dimensão sobre TP.

---

## 6. Fluxo de Dados

```
questionario_real.csv
        │
        ▼
formatar_dataset_real() → df_real (padronizado)
        │
        ├──► calcular_target_probs() → distribuição Likert por item
        │
        ▼
gerar_personas_df() → personas (N perfis demográficos)
        │
        ▼
responder_persona() [LLM] → respostas Likert + justificativa (por persona × sessão)
        │
        ▼
outputs/
├── hdt_personas.csv
├── hdt_respostas_7s.csv
├── questionario_sintetico_N.csv
└── hdt_personas_respostas.csv
```

---

## 7. Dependências

- `openai` / `ollama` — comunicação com LLMs
- `gradio` — interface web
- `pandas` / `numpy` — manipulação de dados
- `matplotlib` — gráficos comparativos
- `certifi` — certificados SSL
- `tqdm` — barra de progresso (modo batch)
- `python-dotenv` — carregamento de variáveis de ambiente

---

## 8. Como Executar

```bash
pip install openai ollama gradio numpy pandas matplotlib certifi tqdm python-dotenv
```

Abrir e executar `00_ui_gerar_dados_sinteticos.ipynb`. A interface Gradio será servida em `http://127.0.0.1:7860`.
