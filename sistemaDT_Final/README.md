# sistemaDT — Digital Twin Cognitivo-Organizacional da Proteção Tecnológica

Implementação da **SPEC v2.1** (Fases 0+1): gêmeo digital de Fatores Humanos em
Cibersegurança com motor **PLS-SEM** (framework 7S → Proteção Tecnológica),
calibrado com o estudo real (N=142) e validado contra os exports do SmartPLS.

> **Integridade científica:** todo dado gerado pelo sistema é **sintético**,
> carimbado como tal (coluna `__synthetic`, sidecar de proveniência e aviso nos
> exports). O modelo é correlacional; what-ifs são projeções sob o modelo
> estimado, não inferência causal experimental.

## Início rápido

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/seed.py          # calibra o gêmeo com a onda 1 (dados/onda1)
.venv/bin/uvicorn twin_api.main:app      # painel em http://localhost:8000
```

Usuários de demonstração (troque via `TWIN_USERS` fora do ambiente local):

| Usuário | Senha | Papel |
|---|---|---|
| `admin` | `admin123` | admin (calibra modelos) |
| `analista` | `analista123` | analyst (gera, simula, decide recomendações) |
| `leitor` | `leitor123` | viewer (somente leitura) |

Docker: `docker compose up --build` (mesma porta 8000).

## O que está implementado (Fase 1)

| RF | Entrega |
|---|---|
| RF1 Calibração (2 caminhos) | `POST /models` (CSV bruto, estimador PLS leve próprio) **ou** `POST /models/smartpls` (exports do SmartPLS: cross loadings + Fornell-Larcker + descritivas [+ correlações], β/R² do diagrama) — no 2º caminho o what-if roda sobre o modelo aferido pelo SmartPLS, sem reestimação; bootstrap/equivalência exigem o bruto |
| RF2 Geração sintética | `POST /models/{id}/generate` — cópula gaussiana (marginais empíricas), ruído humano (desatentos, straightlining, aquiescência), semente reprodutível |
| RF3 Bateria psicométrica | `GET /models/{id}/battery` + `GET .../bootstrap` — α, rho_A, CR, AVE, HTMT, VIF, f², caminhos, R², significância |
| RF5 What-if / Monte Carlo | `POST /models/{id}/simulate`, `POST /scenarios` + `/run` — shift/set (semântica na SPEC 6.11), war-gaming |
| RF6 Sensibilidade | `GET /models/{id}/sensitivity` (tornado, IPMA) + recomendações com IC |
| RF8 Equivalência | `POST /models/{id}/equivalence` — descritivos, congruência de Tucker, MGA por permutação, MICOM passo 2 |
| RF10 Painel | SPA em `/` — Visão Geral, Preditivo, Prescritivo, Validações, Governança |

Perfil piloto (SPEC 10.4): SQLite + execução síncrona + JWT/RBAC local.
`DATABASE_URL` troca para Postgres sem mudança de código.

## Validação contra o SmartPLS (CA2)

`tests/test_golden_smartpls.py` compara o motor com os exports reais do estudo:
α exato (3 casas), HTMT/VIF exatos, cargas/CR/AVE/rho_A/Φ/β/R² dentro de 0.02.

```bash
.venv/bin/pytest                                  # 32 testes
.venv/bin/pytest --cov=twin_core                  # cobertura do core (~94%; NF8 >= 85%)
```

## Estrutura

```
twin_core/    motor científico (PLS, geração, simulação, psicometria, equivalência)
twin_api/     API FastAPI (JWT+RBAC, SQLite/Postgres, endpoints Fase 1)
twin_web/     painel SPA (5 abas, selo de dados sintéticos)
dados/onda1/  CSV bruto (N=142) + exports SmartPLS (alvos de validação)
scripts/      seed.py (calibração inicial)
tests/        golden tests SmartPLS + core + API
```

## Roadmap

Fase 2 (ondas múltiplas, drift, RF4/RF7/RF9) e Fases 3–4 (camada agêntica,
copiloto RAG, RL) conforme SPEC v2.1 Seção 17.
