# Relatório de Implementação — sistemaDT (Digital Twin Cognitivo-Organizacional da Proteção Tecnológica)

## 1. Visão Geral

O `sistemaDT_Final` implementa a **SPEC v2.1 (Fases 0+1)**: um gêmeo digital de Fatores Humanos em Cibersegurança com motor **PLS-SEM** (framework 7S → Proteção Tecnológica), calibrado com o estudo real (N=142) e validado contra os exports do SmartPLS.

O sistema é composto por três camadas:

| Camada | Descrição |
|--------|-----------|
| `twin_core` | Motor científico: PLS, geração sintética, simulação, psicometria, equivalência |
| `twin_api` | API FastAPI com JWT/RBAC, persistência SQLite/Postgres, endpoints Fase 1 |
| `twin_web` | Painel SPA com 5 abas e selo de dados sintéticos |

> **Integridade científica:** todo dado gerado é **sintético**, carimbado com coluna `__synthetic`, sidecar de proveniência e aviso nos exports. O modelo é correlacional; what-ifs são projeções sob o modelo estimado, não inferência causal experimental.

---

## 2. Arquitetura

```
sistemaDT_Final/
├── twin_core/        motor científico (PLS, geração, simulação, psicometria, equivalência)
├── twin_api/         API FastAPI (JWT+RBAC, SQLite/Postgres, endpoints Fase 1)
│   └── routes/       rotas de modelos, simulação e cenários
├── twin_web/         painel SPA (5 abas, selo de dados sintéticos)
├── dados/onda1/      CSV bruto (N=142) + exports SmartPLS (alvos de validação)
├── scripts/          seed.py (calibração inicial)
├── tests/            golden tests SmartPLS + core + API
├── var/              banco SQLite, datasets e ondas gerados
├── pyproject.toml    dependências e configuração do projeto
├── Dockerfile        imagem Docker
└── docker-compose.yml
```

---

## 3. Motor Científico — `twin_core`

### 3.1 Instrumento (`instrument.py`)

Define a estrutura canônica do modelo 7S → Proteção Tecnológica:

| Construto | Código | Tipo | Nome completo |
|-----------|--------|------|---------------|
| Shared Values | SV | Exógeno | Valores Compartilhados |
| Strategy | SG | Exógeno | Estratégia |
| Structure | SU | Exógeno | Estrutura |
| Systems | SM | Exógeno | Sistemas |
| Staff | SF | Exógeno | Equipe |
| Style | SY | Exógeno | Estilo de Liderança |
| Skills | SK | Exógeno | Habilidades |
| Technology Protection | TP | Endógeno | Proteção Tecnológica |

Cada construto tem 4 itens (SV1–SV4, SG1–SG4, ..., TP1–TP4), totalizando 32 indicadores reflexivos.

### 3.2 Modelo Calibrado (`model.py`)

O dataclass `CalibratedModel` é o estado paramétrico do gêmeo, contendo:

| Campo | Descrição |
|-------|-----------|
| `loadings` | Cargas fatoriais λ por item |
| `weights` | Pesos externos por construto (escore com DP=1) |
| `beta` | Coeficientes de caminho exógeno → TP |
| `r2` | Coeficiente de determinação do modelo estrutural |
| `phi` | Matriz de correlações latentes entre todos os construtos |
| `means` / `sds` | Média e DP de cada item na escala original (1–7) |
| `item_corr` | Matriz empírica 32×32 de correlações entre itens (opcional) |
| `item_freqs` | Distribuição empírica 1–7 por item (usada na cópula gaussiana) |
| `noise` | Parâmetros de ruído comportamental humano |
| `provenance` | Metadados de calibração (fonte, N, data, estimador) |

O modelo é serializado em JSON com hash SHA-256 de proveniência para rastreabilidade.

### 3.3 Estimador PLS-SEM (`pls.py`)

Implementa o estimador PLS leve (modo A, esquema de ponderação de caminhos), reproduzindo o SmartPLS dentro de tolerância (validado pelos golden tests):

1. **Iteração interna/externa**: itera pesos externos (modo A) com esquema de caminhos até convergência (tolerância 1e-8, máx. 500 iterações).
2. **Escores latentes**: padronizados (média 0, DP amostral 1).
3. **Cargas**: correlação de cada item com o escore do próprio construto.
4. **Estrutural**: OLS dos escores exógenos sobre o endógeno (TP).
5. **Bootstrap** (`bootstrap_paths`): 2000 reamostras de linhas com reestimação completa; retorna estimativa, SE, t, p-valor e IC 95% por caminho.

### 3.4 Calibração (`calibrate.py`)

Dois caminhos de calibração (RF1):

**Caminho 1 — CSV bruto** (`calibrate_from_responses`):
- Lê o DataFrame de respostas, detecta colunas de item por prefixo.
- Estima o modelo PLS completo.
- Calcula bateria psicométrica completa (RF3).
- Armazena matriz de correlação empírica e distribuição de frequências por item (habilita geração por cópula gaussiana).

**Caminho 2 — Exports SmartPLS** (`calibrate_from_smartpls`):
- Importa cross loadings, Fornell-Larcker, descritivas, β e R² do diagrama.
- O what-if roda sobre os parâmetros aferidos pelo SmartPLS, sem reestimação.
- Bootstrap e equivalência ficam indisponíveis (sem microdado).
- Opcional: matriz de correlações dos itens (habilita bateria psicométrica exata).

### 3.5 Geração Sintética (`generate.py`)

O método `generate_sample` gera N respondentes sintéticos (RF2) em dois modos:

**Modo `model`** (padrão):
1. Amostra latentes exógenos ~ MVN(0, Φ) com intervenções opcionais.
2. Propaga η_TP = escala·(β'η) + √(1−R²)·ζ.
3. Gera itens via cargas: z_i = λ_i·η_c + √(1−λ²)·ε.
4. Mapeia z ~ N(0,1) para escala 1–7 por **cópula gaussiana** (quantile matching na distribuição empírica).

**Modo `empirical`**:
- Amostra itens diretamente da MVN com a matriz de correlação empírica 32×32.
- Maior fidelidade item a item; não suporta intervenções.

Em ambos os modos, aplica **ruído comportamental humano** (`noise.py`):

| Tipo | Parâmetro | Comportamento |
|------|-----------|---------------|
| Desatentos | `p_careless = 0.07` | Respostas uniformes aleatórias 1–7 |
| Straightlining | `p_straight = 0.05` | Mesmo valor em todos os itens |
| Aquiescência | `sigma_acq = 0.35` | Viés individual somado a todos os itens |

O dataset gerado inclui colunas `__careless`, `__straightline` e `__synthetic = True`, além de um sidecar JSON de proveniência.

### 3.6 Simulação What-if / Monte Carlo (`simulate.py`)

Implementa RF5 com dois tipos de intervenção:

| Tipo | Semântica |
|------|-----------|
| `shift` | Soma delta (DP) ao latente, preservando correlações |
| `set` | Fixa o latente em valor absoluto (operador do) |

O método `monte_carlo_whatif` executa K réplicas de N respondentes e retorna:
- ΔTP médio em DP latente, pontos do índice (1–7) e escala IPMA (0–100).
- IC 95% por percentil.
- P(ΔTP > 0).

Inclui o cenário exemplo `scenario_wargame_phishing`: choque em Habilidades (SK, −0.6 DP) e Equipe (SF, −0.4 DP).

### 3.7 Bateria Psicométrica (`psychometrics.py`)

Implementa RF3 com as métricas do SmartPLS 4:

| Métrica | Descrição |
|---------|-----------|
| α (Cronbach padronizado) | k·r̄ / (1 + (k−1)·r̄) |
| rho_A (Dijkstra-Henseler) | Confiabilidade composta alternativa |
| rho_c (CR) | (Σλ)² / ((Σλ)² + Σ(1−λ²)) |
| AVE | Média das cargas ao quadrado |
| HTMT | Heterotraço-monotraço entre pares de construtos |
| VIF externo | 1/(1−R²) do item regredido nos demais do bloco |
| VIF interno | Entre preditores (a partir de Φ dos exógenos) |
| f² | Tamanho de efeito por preditor |

### 3.8 Sensibilidade e IPMA (`sensitivity.py`)

Implementa RF6:

- **Efeitos marginais**: ∂TP/∂η_c = escala × β_c (modelo linear padronizado).
- **Tornado**: ΔTP em pontos (1–7) para shift de ±1 DP em cada construto, ordenado por |efeito|.
- **IPMA de construtos**: importância (β) × desempenho (média reescalada 0–100).
- **IPMA de itens**: importância = peso externo não padronizado × β do construto; prioridade = importância × folga (100 − desempenho).
- **Recomendações** (`recommend`): top-N construtos por importância × folga, com efeito esperado e IC 95% via Monte Carlo. Cada recomendação inclui itens-alavanca (2º estágio IPMA) e flag `requires_human_approval = True`.

### 3.9 Equivalência (`equivalence.py`)

Implementa RF8 — bateria de equivalência real ↔ sintético:

| Teste | Critério de aprovação |
|-------|-----------------------|
| Descritivos | Δ médio máximo < 0.5 pontos |
| Congruência de Tucker | Coeficiente > 0.95 por construto |
| MGA por permutação | p > 0.05 em todos os caminhos (H0: caminhos iguais) |
| MICOM passo 2 | c acima do quantil 5% da distribuição nula (invariância composicional) |

---

## 4. API REST — `twin_api`

### 4.1 Autenticação e RBAC (`security.py`)

JWT (HS256) com TTL configurável (`TWIN_TOKEN_TTL_S`, padrão 8h). Três papéis hierárquicos:

| Papel | Permissões |
|-------|-----------|
| `viewer` | Leitura (modelos, bateria, baseline, sensibilidade, datasets) |
| `analyst` | viewer + geração, simulação, equivalência, recomendações |
| `admin` | analyst + calibração de modelos |

Usuários configuráveis via variável de ambiente `TWIN_USERS` (`user:senha:papel,...`).

### 4.2 Endpoints de Modelos (`routes/models.py`)

| Método | Rota | RF | Descrição |
|--------|------|----|-----------|
| POST | `/models` | RF1 | Calibra a partir do CSV bruto |
| POST | `/models/smartpls` | RF1 | Calibra a partir dos exports SmartPLS |
| GET | `/models` | — | Lista modelos |
| GET | `/models/{id}` | — | Detalhes do modelo |
| GET | `/models/{id}/battery` | RF3 | Bateria psicométrica |
| GET | `/models/{id}/baseline` | — | Estado atual dos construtos |
| GET | `/models/{id}/bootstrap` | RF3 | Significância dos caminhos por bootstrap |
| GET | `/models/{id}/sensitivity` | RF6 | Efeitos marginais, tornado, IPMA |
| POST | `/models/{id}/generate` | RF2 | Gera dataset sintético carimbado |
| POST | `/models/{id}/recommendations` | RF6 | Emite recomendações priorizadas |
| GET | `/models/{id}/recommendations` | RF6 | Lista recomendações |
| GET | `/models/{id}/recommendations/report.pdf` | RF11 | PDF das recomendações aceitas |
| PATCH | `/recommendations/{id}` | — | Registra decisão humana (aceitar/rejeitar) |

### 4.3 Endpoints de Simulação (`routes/simulate.py`)

| Método | Rota | RF | Descrição |
|--------|------|----|-----------|
| POST | `/models/{id}/simulate` | RF5 | Monte Carlo baseline ou what-if direto |
| POST | `/scenarios` | RF5 | Cria cenário persistido |
| GET | `/scenarios` | RF5 | Lista cenários |
| POST | `/scenarios/{id}/run` | RF5 | Executa cenário |
| GET | `/simulations/{id}` | RF5 | Resultado de simulação |
| POST | `/models/{id}/equivalence` | RF8 | Bateria de equivalência real × sintético |

### 4.4 Persistência (`db.py`, `orm.py`)

SQLite por padrão (`var/twin.db`); troca para Postgres via `DATABASE_URL` sem mudança de código. Entidades principais:

| Tabela | Conteúdo |
|--------|----------|
| `ModelRow` | Parâmetros JSON, bateria, hash de proveniência, caminho da onda bruta |
| `DatasetRow` | Datasets sintéticos gerados (caminho CSV + sidecar JSON) |
| `ScenarioRow` | Cenários de war-gaming persistidos |
| `SimulationRow` | Resultados de simulações Monte Carlo |
| `RecommendationRow` | Recomendações emitidas com status (emitted/accepted/rejected/archived) |
| `AuditLogRow` | Trilha de auditoria de todas as ações (ator, ação, entidade, detalhe) |

---

## 5. Painel Web — `twin_web`

SPA em HTML/CSS/JS com 5 abas:

| Aba | Conteúdo |
|-----|----------|
| Visão Geral | Baseline dos construtos, R², coeficientes β |
| Preditivo | Geração de datasets sintéticos (RF2) |
| Prescritivo | Simulação what-if e recomendações (RF5/RF6) |
| Validações | Bateria psicométrica e equivalência (RF3/RF8) |
| Governança | Trilha de auditoria e decisões humanas |

Todos os dados sintéticos exibidos carregam o **selo de dados sintéticos** conforme SPEC Seção 13.

---

## 6. Calibração Inicial — `scripts/seed.py`

Semeia o gêmeo com a onda 1 real (`dados/onda1/respostas_full.csv`):

```bash
python scripts/seed.py [--nome modelo-7s-tp] [--csv caminho/para/respostas.csv]
```

O script:
1. Lê o CSV de respostas e os rótulos dos itens (cross loadings).
2. Calibra o modelo via `calibrate_from_responses`.
3. Verifica se o modelo já foi semeado (por hash de proveniência).
4. Persiste o `ModelRow` no banco e copia o CSV da onda para `var/waves/`.

---

## 7. Testes

```bash
.venv/bin/pytest                   # 32 testes
.venv/bin/pytest --cov=twin_core   # cobertura do core (~94%; NF8 >= 85%)
```

| Arquivo de teste | Cobertura |
|-----------------|-----------|
| `test_golden_smartpls.py` | Valida α, HTMT, VIF, cargas, CR, AVE, rho_A, Φ, β, R² contra exports reais do SmartPLS (tolerância 0.02) |
| `test_equivalence.py` | Bateria RF8 (descritivos, Tucker, MGA, MICOM) |
| `test_generate_simulate.py` | Geração sintética e Monte Carlo |
| `test_api.py` | Endpoints FastAPI (calibração, geração, simulação, recomendações) |
| `test_smartpls_import.py` | Importação dos exports SmartPLS (caminho 2 do RF1) |

---

## 8. Fluxo de Dados

```
dados/onda1/respostas_full.csv
        │
        ▼
calibrate_from_responses() → CalibratedModel {λ, β, Φ, médias, DP, R²}
        │
        ├──► full_battery() → bateria psicométrica (α, rho_A, CR, AVE, HTMT, VIF, f²)
        │
        ├──► generate_sample() [cópula gaussiana + ruído humano]
        │         └──► var/datasets/{id}.csv  (carimbado __synthetic=True + sidecar JSON)
        │
        ├──► monte_carlo_whatif() [K réplicas × N respondentes]
        │         └──► ΔTP (DP latente, pontos 1-7, IPMA 0-100) + IC 95%
        │
        ├──► recommend() [IPMA × folga + Monte Carlo]
        │         └──► RecommendationRow (emitted → accepted/rejected → archived)
        │
        └──► equivalence_report() [descritivos, Tucker, MGA, MICOM]
                  └──► verdict: equivalent = True/False
```

---

## 9. Dependências

- `numpy` / `pandas` / `scipy` — motor numérico e estatístico
- `fastapi` / `uvicorn` — API REST assíncrona
- `sqlalchemy` — ORM (SQLite/Postgres)
- `pydantic` — validação de schemas
- `PyJWT` — autenticação JWT
- `fpdf2` — geração de relatórios PDF
- `python-multipart` — upload de arquivos na API

---

## 10. Como Executar

### Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Calibração inicial

```bash
.venv/bin/python scripts/seed.py
```

### API

```bash
.venv/bin/uvicorn twin_api.main:app
# Painel em http://localhost:8000
```

### Docker

```bash
docker compose up --build
# Mesma porta 8000
```

### Usuários de demonstração

| Usuário | Senha | Papel |
|---------|-------|-------|
| `admin` | `admin123` | admin |
| `analista` | `analista123` | analyst |
| `leitor` | `leitor123` | viewer |

Configuráveis via variável de ambiente `TWIN_USERS` (`user:senha:papel,...`).
