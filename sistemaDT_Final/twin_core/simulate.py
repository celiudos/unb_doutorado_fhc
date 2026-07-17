"""RF5 — Intervenções, propagação estrutural e Monte Carlo (what-if / war-gaming)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instrument import EXOGENOUS
from .model import CalibratedModel, ensure_pd_corr


@dataclass
class Intervention:
    """`shift`: soma delta (DP) preservando correlações; `set`: fixa valor (operador do)."""
    construct: str
    kind: str            # "shift" | "set"
    value: float

    def __post_init__(self):
        if self.kind not in ("shift", "set"):
            raise ValueError(f"Tipo de intervenção inválido: {self.kind}")
        if self.construct not in EXOGENOUS:
            raise ValueError(f"Intervenção só é permitida em construto exógeno, recebido: {self.construct}")


def parse_interventions(raw: list[dict]) -> list[Intervention]:
    out = []
    for item in raw:
        kind = item.get("kind") or item.get("tipo")
        value = item.get("value", item.get("delta"))
        out.append(Intervention(construct=item["construct"], kind=str(kind), value=float(value)))
    return out


def sample_exo_latents(
    model: CalibratedModel,
    n: int,
    rng: np.random.Generator,
    interventions: list[Intervention] | None = None,
) -> np.ndarray:
    """Amostra latentes exógenos ~ MVN(0, Φ) e aplica intervenções (semântica da Seção 6.11)."""
    phi = ensure_pd_corr(model.phi_exo()).to_numpy()
    L = rng.multivariate_normal(np.zeros(len(EXOGENOUS)), phi, size=n, method="cholesky")
    for iv in interventions or []:
        j = EXOGENOUS.index(iv.construct)
        if iv.kind == "shift":
            L[:, j] += iv.value
        else:  # set-point / operador do: rompe correlações
            L[:, j] = iv.value
    return L


def propagate_tp(model: CalibratedModel, L_exo: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """η_TP = escala·(β'η) + √(1-R²)·ζ, com escala congelada na linha de base (SPEC 6.11)."""
    beta = model.beta_vector()
    phi = model.phi_exo().to_numpy()
    var_lin = float(beta @ phi @ beta)          # variância do preditor linear na linha de base
    scale = np.sqrt(model.r2 / var_lin) if var_lin > 0 else 0.0
    lin = L_exo @ beta
    zeta = rng.normal(0, 1, size=len(L_exo))
    return scale * lin + np.sqrt(max(0.0, 1 - model.r2)) * zeta


def monte_carlo_whatif(
    model: CalibratedModel,
    interventions: list[Intervention] | None = None,
    n: int = 200,
    k: int = 5000,
    seed: int = 42,
) -> dict:
    """K réplicas de N respondentes: ΔTP médio vs linha de base, IC 95%, P(ΔTP>0).

    ΔTP é reportado em DP latente e convertido para pontos do índice TP (escala 1-7)
    e para a escala 0-100 do IPMA.
    """
    rng = np.random.default_rng(seed)
    deltas = np.empty(k)
    base_means = np.empty(k)
    for r in range(k):
        Lb = sample_exo_latents(model, n, rng)
        tp_base = propagate_tp(model, Lb, rng)
        Li = sample_exo_latents(model, n, rng, interventions)
        tp_int = propagate_tp(model, Li, rng)
        base_means[r] = tp_base.mean()
        deltas[r] = tp_int.mean() - tp_base.mean()

    factor = model.tp_scale_factor()            # DP latente -> pontos do índice TP (1-7)
    d_mean = float(deltas.mean())
    lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    tp_items = model.blocks[model.endogenous]
    tp_index_baseline = float(np.mean([model.means[i] for i in tp_items]))
    return {
        "n": n, "k": k, "seed": seed,
        "interventions": [vars(iv) for iv in interventions or []],
        "delta_tp_latent": {"mean": d_mean, "ci95": [lo, hi]},
        "delta_tp_points": {"mean": d_mean * factor, "ci95": [lo * factor, hi * factor]},
        "delta_tp_ipma": {"mean": d_mean * factor / 6 * 100,
                          "ci95": [lo * factor / 6 * 100, hi * factor / 6 * 100]},
        "tp_index_baseline": tp_index_baseline,
        "tp_index_projected": tp_index_baseline + d_mean * factor,
        "p_delta_positive": float((deltas > 0).mean()),
        "is_synthetic": True,
    }


def scenario_wargame_phishing(intensity: float = 1.0) -> list[Intervention]:
    """Cenário exemplo da SPEC: campanha de phishing como choque em Habilidades/Equipe."""
    return [
        Intervention("SK", "shift", -0.6 * intensity),
        Intervention("SF", "shift", -0.4 * intensity),
    ]


def baseline_summary(model: CalibratedModel) -> dict:
    """Modo descritivo: estado atual dos construtos (médias 1-7 e escala IPMA 0-100)."""
    out = {}
    for c in model.constructs:
        items = model.blocks[c]
        mean = float(np.mean([model.means[i] for i in items]))
        out[c] = {"mean_1_7": mean, "ipma_0_100": (mean - 1) / 6 * 100}
    return out
