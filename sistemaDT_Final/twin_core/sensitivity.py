"""RF6 — Sensibilidade, tornado, IPMA e recomendação de intervenção priorizada."""

from __future__ import annotations

import numpy as np

from .instrument import CONSTRUCT_NAMES, EXOGENOUS
from .model import CalibratedModel
from .simulate import Intervention, monte_carlo_whatif


def marginal_effects(model: CalibratedModel) -> dict[str, float]:
    """∂TP/∂η_c: no modelo linear padronizado, o próprio β (com a escala congelada)."""
    beta = model.beta_vector()
    phi = model.phi_exo().to_numpy()
    var_lin = float(beta @ phi @ beta)
    scale = np.sqrt(model.r2 / var_lin) if var_lin > 0 else 0.0
    return {c: float(scale * model.beta[c]) for c in EXOGENOUS}


def tornado(model: CalibratedModel, delta: float = 1.0) -> list[dict]:
    """ΔTP (pontos do índice 1-7) para shift de ±delta DP em cada construto, ordenado por |efeito|."""
    eff = marginal_effects(model)
    factor = model.tp_scale_factor()
    rows = []
    for c in EXOGENOUS:
        rows.append({
            "construct": c,
            "name": CONSTRUCT_NAMES[c],
            "delta_up": eff[c] * delta * factor,
            "delta_down": -eff[c] * delta * factor,
        })
    rows.sort(key=lambda r: abs(r["delta_up"]), reverse=True)
    return rows


def ipma(model: CalibratedModel) -> list[dict]:
    """Importância (efeito total = β) x desempenho (0-100), como no SmartPLS."""
    out = []
    for c in EXOGENOUS:
        items = model.blocks[c]
        mean = float(np.mean([model.means[i] for i in items]))
        out.append({
            "construct": c,
            "name": CONSTRUCT_NAMES[c],
            "importance": model.beta[c],
            "performance": (mean - 1) / 6 * 100,
        })
    return out


def ipma_items(model: CalibratedModel) -> list[dict]:
    """IPMA no nível de indicador (o 'IPMA2' do SmartPLS).

    Importância do item = peso externo não padronizado (w_std/DP), normalizado
    dentro do bloco, x efeito total do construto em TP. Desempenho = média
    reescalada 0-100. Itens de construtos com β negativo herdam importância negativa.
    """
    out = []
    for c in EXOGENOUS:
        items = model.blocks[c]
        w_unstd = np.array([model.weights[c][k] / model.sds[i] for k, i in enumerate(items)])
        w_norm = w_unstd / w_unstd.sum()
        for k, i in enumerate(items):
            out.append({
                "item": i,
                "construct": c,
                "construct_name": CONSTRUCT_NAMES[c],
                "label": model.item_labels.get(i, i),
                "importance": float(w_norm[k] * model.beta[c]),
                "performance": (model.means[i] - 1) / 6 * 100,
            })
    return out


def item_levers(model: CalibratedModel, construct: str, top: int = 3) -> list[dict]:
    """Itens-alavanca de um construto (2º estágio do IPMA, Ringle & Sarstedt 2016).

    Prioriza por importância x folga (100 - desempenho) e estima o benefício
    concreto: ΔTP em pontos (1-7) se a média do item subir 1 ponto na escala.
    """
    eff = marginal_effects(model)[construct]   # ∂TP/∂η_c em DP latente (escala congelada x β)
    items = model.blocks[construct]
    rows = []
    for k, i in enumerate(items):
        w_unstd = model.weights[construct][k] / model.sds[i]   # Δη_c por +1 ponto no item
        performance = (model.means[i] - 1) / 6 * 100
        importance = float(w_unstd / sum(model.weights[construct][j] / model.sds[it]
                                         for j, it in enumerate(items)) * model.beta[construct])
        rows.append({
            "item": i,
            "label": model.item_labels.get(i, i),
            "performance": performance,
            "importance": importance,
            # +1 ponto no item -> Δη_c = w_unstd -> ΔTP_latente = eff·w_unstd -> pontos via fator TP
            "delta_tp_per_point": float(eff * w_unstd * model.tp_scale_factor()),
            "priority": importance * (100 - performance),
        })
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows[:top]


def recommend(
    model: CalibratedModel,
    delta: float = 0.5,
    top: int = 3,
    n: int = 200,
    k: int = 2000,
    seed: int = 42,
    exclude: set[str] | None = None,
) -> list[dict]:
    """Recomendações priorizadas: maior efeito marginal positivo com menor desempenho atual.

    Prioridade IPMA: importância x folga de desempenho (headroom). Cada recomendação
    vem com efeito esperado e IC 95% via Monte Carlo (humano decide — Seção 7).

    `exclude`: construtos já em jogo (pendentes/aceitos/rejeitados) que não devem
    reaparecer — a emissão vira uma lista de trabalho sem repetição.
    """
    exclude = exclude or set()
    perf = {r["construct"]: r["performance"] for r in ipma(model)}
    eff = marginal_effects(model)
    ranked = sorted(
        (c for c in EXOGENOUS if eff[c] > 0 and c not in exclude),
        key=lambda c: eff[c] * (100 - perf[c]),
        reverse=True,
    )[:top]

    out = []
    for c in ranked:
        mc = monte_carlo_whatif(model, [Intervention(c, "shift", delta)], n=n, k=k, seed=seed)
        out.append({
            "construct": c,
            "name": CONSTRUCT_NAMES[c],
            "intervention": {"kind": "shift", "value": delta},
            "rationale": (
                f"β={model.beta[c]:+.3f}, desempenho atual {perf[c]:.0f}/100 — "
                f"maior impacto marginal em TP com folga de melhoria"
            ),
            "expected_delta_tp_points": mc["delta_tp_points"]["mean"],
            "ci95_points": mc["delta_tp_points"]["ci95"],
            "p_positive": mc["p_delta_positive"],
            "levers": item_levers(model, c, top=2),   # 2º estágio IPMA: onde atuar na prática
            "requires_human_approval": True,
        })
    return out
