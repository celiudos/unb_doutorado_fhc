# Relatório de Implementação — Gerador de Personas Sintéticas (HDT 7S)

## 1. Visão Geral

A aplicação gera dados sintéticos de pesquisa sobre **proteção tecnológica organizacional** utilizando o conceito de **Human Digital Twins (HDTs)** combinado com o **modelo 7S de McKinsey**. O objetivo é produzir respostas realistas a um questionário Likert (1–7) a partir de personas sintéticas cujos perfis demográficos refletem a distribuição de um dataset real.

---

## 2. Pipeline de Geração de Dados Sintéticos (`01_gerar_dados_sinteticos.ipynb`)

O notebook segue um pipeline sequencial de 6 etapas:

### 2.1 Configurações Gerais

A primeira célula importa `GerarDadosUtils` e instancia o objeto com os parâmetros de configuração:

```python
from utils import GerarDadosUtils

SEED = 84
N_PERSONAS = 1
N_SESSOES_POR_PERSONA = 1
LLM_PROVIDER = "ollama"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b"
OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"

utils = GerarDadosUtils(
    seed=SEED,
    n_personas=N_PERSONAS,
    n_sessoes_por_persona=N_SESSOES_POR_PERSONA,
    llm_provider=LLM_PROVIDER,
    ollama_host=OLLAMA_HOST,
    ollama_model=OLLAMA_MODEL,
    openai_model=OPENAI_MODEL,
)
```

O construtor de `GerarDadosUtils` aceita dois provedores LLM (**Ollama** local ou **OpenAI** via API). Também configura atributos demográficos (idade, sexo, setor, porte, cargo, tempo de atuação, treinamento) com proporções padrão derivadas dos dados reais, todas sobrescrevíveis.

### 2.2 Configuração do LLM

O método `get_response` despacha o prompt para o provedor selecionado:

- **Ollama**: usa `ollama.Client` com parâmetros de seed, temperatura e `format="json"`.
- **OpenAI**: usa `openai.OpenAI` com temperatura configurável.

Um teste rápido de conexão pode ser feito com:
```python
utils.get_response("Olá, qual é a capital da França?", temperature=0.5)
```

### 2.3 Ajustando Dados Reais e Gerando Personas

#### Formatação do dataset real

```python
df_real = utils.formatar_dataset_real("inputs/questionario_real.csv")
df_real.to_csv("inputs/questionario_real_formatado.csv", index=False)
```

O método `formatar_dataset_real`:
1. Lê o CSV (detecta formato padrão ou legado).
2. Converte colunas numéricas (itens Likert, idade, tempo).
3. Limpa strings categóricas.
4. Remove linhas completamente vazias.

Resultado: DataFrame padronizado com 32 itens Likert + 7 atributos demográficos.

#### Geração de personas

```python
personas_df = utils.gerar_personas_df()
personas_df = utils.sobrescrever_com_distribuicao_real(personas_df, df_real)
```

O método `gerar_personas_df` cria N personas com atributos demográficos amostrados por **proporções ponderadas** (`random.choices`). Em seguida, `sobrescrever_com_distribuicao_real` substitui os atributos gerados por valores amostrados diretamente da distribuição empírica do dataset real, garantindo fidelidade estatística.

#### Estatísticas descritivas

O notebook calcula e exibe uma tabela com contagem e percentual por atributo/categoria (idade em faixas, sexo, setor, porte, cargo, tempo de atuação, treinamento), permitindo verificar se a distribuição das personas geradas é coerente com os dados reais.

### 2.4 Questionário do Modelo 7S

O questionário é definido como uma lista de 32 dicionários em `GerarDadosUtils.QUESTIONARIO_7S`, cada um com:

| Campo | Descrição |
|-------|-----------|
| `item` | Código (ex: SV1, SG2, TP4) |
| `dimensao` | Dimensão do modelo 7S |
| `tipo_variavel` | `independente` ou `dependente` |
| `afirmacao` | Texto da afirmação em português |

As 8 dimensões são:

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

### 2.5 Instanciação dos HDTs e Geração de Respostas

#### Cálculo da distribuição-alvo

```python
target_probs = utils.calcular_target_probs(df_real)
```

Para cada item do questionário, calcula a frequência relativa de cada valor Likert (1–7) no dataset real. Essa distribuição é incluída no prompt para guiar o LLM a gerar respostas com distribuição estatística semelhante à real.

#### Montagem do prompt HDT

O método `montar_prompt_hdt` constrói um prompt contendo:
1. Instrução de role-play como Human Digital Twin brasileiro.
2. Perfil demográfico completo da persona (JSON).
3. As 32 afirmações do questionário com dimensão e código.
4. Escala Likert explicada (1–7).
5. Regras de variabilidade e formato de saída (JSON estrito).
6. Distribuição-alvo por item para calibração estatística.

#### Geração de respostas por persona

O método `responder_persona`:
1. Monta o prompt com perfil + distribuição-alvo.
2. Calcula uma seed determinística por persona/sessão.
3. Envia ao LLM com `temperature=0.8`.
4. Extrai o JSON da resposta (com fallback via regex).
5. Normaliza as respostas em um dicionário plano.

### 2.6 Execução das Sessões e Salvamento

```python
resultados = []
for sessao in range(1, N_SESSOES_POR_PERSONA + 1):
    for persona in tqdm(personas, desc="Gerando respostas"):
        linha = utils.responder_persona(persona, target_probs, sessao=sessao)
        linha["sessao"] = sessao
        resultados.append(linha)

df_respostas = pd.DataFrame(resultados)
```

O loop itera sobre sessões × personas, chamando o LLM para cada combinação. Erros são capturados e logados sem interromper o pipeline.

#### Artefatos salvos

| Arquivo | Conteúdo |
|---------|----------|
| `outputs/hdt_personas.csv` | Perfis demográficos das personas |
| `outputs/questionario_7s.csv` | Definição do questionário |
| `outputs/hdt_respostas_7s.csv` | Respostas completas com metadados |
| `inputs/questionario_sintetico_N.csv` | Formato exportável (Perfil, sexo, idade + itens) |

---

## 3. Comparação entre Dados Sintéticos e Reais (`02_comparacao_sintetico_real.ipynb`)

Este notebook avalia a **human-likeness** dos Human Digital Twins comparando distribuições de resposta, estatísticas descritivas por dimensão e equivalência estatística.

### 3.1 Objetivo da Validação

Validar se os HDTs são suficientemente parecidos com os humanos para entrar no modelo PLS-SEM. As métricas utilizadas são:

1. **Mann-Whitney U**: compara as distribuições item a item. `p > 0,05` indica sem diferença significativa.
2. **Cohen's d**: mede a magnitude prática das diferenças. `|d| < 0,20` = efeito negligenciável; `|d| < 0,50` = efeito pequeno.
3. **Diferença média Likert**: distância média entre notas reais e sintéticas.
4. **Correlação de Spearman**: avalia se ambos os grupos ordenam os itens de forma parecida (semelhança no ranking).

### 3.2 Carregamento dos Dados

```python
df_real = pd.read_csv("inputs/questionario_real_formatado.csv")
df_sint = pd.read_csv("inputs/questionario_sintetico_1000.csv")
```

O notebook carrega o dataset real formatado e o dataset sintético gerado (ex: 1000 respondentes).

### 3.3 Estatísticas Descritivas por Dimensão

Para cada dimensão do modelo 7S, calcula-se:
- Média e desvio-padrão dos dados reais.
- Média e desvio-padrão dos dados sintéticos.
- Diferença entre as médias (Δ Média).

Exemplo de resultado:

| Dimensão | Média Real | DP Real | Média Sint | DP Sint | Δ Média |
|----------|-----------|---------|-----------|---------|---------|
| Shared Values | 5.804 | 1.389 | 5.938 | 0.787 | 0.133 |
| Strategy | 5.179 | 1.586 | 5.299 | 0.788 | 0.119 |
| Technology Protection | 4.810 | 1.827 | 4.580 | 0.687 | -0.229 |

### 3.4 Testes de Equivalência Estatística

Para cada um dos 32 itens:
1. Aplica-se o teste **Mann-Whitney U** (bicaudal).
2. Calcula-se o **Cohen's d** (pooled).
3. Classifica-se o efeito: negligenciável (`|d| < 0.2`), pequeno (`|d| < 0.5`), médio/grande.

Critérios de sucesso:
- Itens SEM diferença significativa (p > 0.05): quanto mais, melhor.
- Itens com efeito negligenciável: quanto mais, melhor.

### 3.5 Distribuição de Frequências Likert por Dimensão

Gera gráficos de barras agrupadas (Real vs. Sintético) para cada dimensão, mostrando a frequência percentual de cada valor Likert (1–7). Permite visualizar se o LLM reproduziu adequadamente a forma da distribuição real.

### 3.6 Médias por Dimensão — Real vs. Sintético

Gráfico de barras agrupadas comparando a média Likert por dimensão entre dados reais e sintéticos, com linha de referência no ponto neutro (4).

### 3.7 Correlação de Spearman (Human-Likeness Global)

```python
medias_r = [df_real[item].mean() for item in ITENS]
medias_s = [df_sint[item].mean() for item in ITENS]
rho, p = stats.spearmanr(medias_r, medias_s)
```

A correlação de Spearman entre as médias por item dos dois grupos mede se os itens que os humanos avaliaram como mais altos também foram avaliados como mais altos pelos HDTs. Um valor alto (ex: ρ = 0.8) indica boa semelhança no ranking das respostas.

---

## 4. Arquitetura do Código

```
00_ui_gerar_dados_sinteticos.ipynb   ← Interface Gradio (UI)
01_gerar_dados_sinteticos.ipynb      ← Pipeline de geração (modo notebook)
02_comparacao_sintetico_real.ipynb    ← Validação estatística
utils/
├── UIUtils.py                       ← Interface Gradio (apresentação e orquestração)
├── GerarDadosUtils.py               ← Lógica de negócio (geração via LLM)
└── Sobre.md                         ← Documentação exibida na UI
inputs/
└── questionario_real.csv            ← Dataset real de referência
outputs/                             ← CSVs e gráficos gerados
```

---

## 5. Dependências

- `openai` / `ollama` — comunicação com LLMs
- `gradio` — interface web
- `pandas` / `numpy` — manipulação de dados
- `matplotlib` — gráficos comparativos
- `scipy` — testes estatísticos (Mann-Whitney, Spearman)
- `certifi` — certificados SSL
- `tqdm` — barra de progresso (modo batch)
- `python-dotenv` — carregamento de variáveis de ambiente

---

## 6. Como Executar

```bash
pip install openai ollama gradio numpy pandas matplotlib certifi tqdm python-dotenv
```

### Geração via notebook
Abrir e executar `01_gerar_dados_sinteticos.ipynb`.

### Geração via interface
Abrir e executar `00_ui_gerar_dados_sinteticos.ipynb`. A interface Gradio será servida em `http://127.0.0.1:7860`.

### Validação
Após gerar os dados sintéticos, executar `02_comparacao_sintetico_real.ipynb` para avaliar a qualidade dos HDTs.
