"""RF1 — Calibração do gêmeo: CSV bruto de respostas OU pacote de parâmetros do SmartPLS."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .instrument import CONSTRUCTS, ENDOGENOUS, EXOGENOUS, blocks_from_items
from .model import CalibratedModel, NoiseParams
from .pls import pls_sem
from .psychometrics import full_battery


def calibrate_from_responses(
    df: pd.DataFrame,
    name: str = "modelo-7s-tp",
    noise: NoiseParams | None = None,
    item_labels: dict[str, str] | None = None,
    source: str = "csv",
) -> tuple[CalibratedModel, dict]:
    """Calibra {λ, β, Φ, médias, DP, R²} do bruto e devolve (modelo, bateria RF3).

    O DataFrame deve ter colunas com códigos de item (SV1..TP4); colunas extras
    (demografia) são ignoradas na estimação e registradas na proveniência.
    """
    item_cols = [c for c in df.columns if isinstance(c, str) and len(c) == 3 and c[:2] in CONSTRUCTS]
    blocks = blocks_from_items(item_cols)
    data = df[sorted(item_cols)].astype(float)

    res = pls_sem(data, blocks, ENDOGENOUS)
    if not res.converged:
        raise RuntimeError("Estimador PLS não convergiu")

    battery = full_battery(data, res)
    item_corr = data.corr()

    model = CalibratedModel(
        name=name,
        constructs=list(blocks),
        endogenous=ENDOGENOUS,
        blocks=blocks,
        loadings=res.loadings,
        weights={c: [float(v) for v in w] for c, w in res.weights.items()},
        beta=res.beta,
        r2=res.r2,
        phi={a: {b: float(res.phi.loc[a, b]) for b in res.constructs} for a in res.constructs},
        means={i: float(data[i].mean()) for i in data.columns},
        sds={i: float(data[i].std(ddof=1)) for i in data.columns},
        item_labels=item_labels or {},
        item_corr={a: {b: float(item_corr.loc[a, b]) for b in data.columns} for a in data.columns},
        item_freqs={
            i: {str(k): float((data[i] == k).mean()) for k in range(1, 8)}
            for i in data.columns
        },
        noise=noise or NoiseParams(),
        provenance={
            "source": source,
            "n": int(len(data)),
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "extra_columns": [c for c in df.columns if c not in item_cols],
            "estimator": "twin-core PLS (modo A, path weighting)",
            "pls_iterations": res.iterations,
        },
    )
    return model, battery


def calibrate_from_smartpls(
    cross_loadings,
    fornell_larcker,
    descriptives,
    beta: dict[str, float],
    r2: float,
    item_correlations=None,
    p_values: dict[str, float] | None = None,
    name: str = "modelo-smartpls",
    noise: NoiseParams | None = None,
    source: str = "smartpls_exports",
) -> tuple[CalibratedModel, dict]:
    """RF1 (2º caminho): calibra o gêmeo direto dos exports do SmartPLS, sem microdado.

    Obrigatórios: cross loadings (λ + wording), Fornell-Larcker (Φ latente) e
    Indicator data original (médias/DPs, N). β e R² vêm do diagrama de caminhos
    (o SmartPLS não os exporta em CSV nesta pasta). Opcional: Indicator data
    correlations (matriz 32x32) — habilita bateria psicométrica exata e o modo
    de geração empírico.

    O what-if resultante roda sobre os parâmetros AFERIDOS PELO SMARTPLS,
    sem reestimação do twin-core no caminho.
    """
    from .io_smartpls import (
        load_cross_loadings,
        load_descriptives,
        load_fornell_larcker,
        load_item_correlations,
        load_item_labels,
    )
    from .psychometrics import (
        ave,
        composite_reliability,
        cronbach_alpha_std,
        htmt_matrix,
        inner_vif,
        outer_vif,
        rho_a,
    )

    if set(beta) != set(EXOGENOUS):
        raise ValueError(f"beta deve ter exatamente os construtos exógenos {EXOGENOUS}")
    if not 0 < r2 < 1:
        raise ValueError("R² deve estar em (0, 1)")

    cl = load_cross_loadings(cross_loadings)
    labels = load_item_labels(cross_loadings)
    phi_full = load_fornell_larcker(fornell_larcker)
    desc = load_descriptives(descriptives)

    items = sorted(i for i in cl.index if i is not None)
    blocks = blocks_from_items(items)
    missing = [i for i in items if i not in desc.index]
    if missing:
        raise ValueError(f"Descritivas não cobrem os itens: {missing}")

    # diagonal do Fornell-Larcker é a raiz do AVE, não 1 — corrige para matriz de correlação
    phi_np = phi_full.loc[CONSTRUCTS, CONSTRUCTS].to_numpy(dtype=float, copy=True)
    np.fill_diagonal(phi_np, 1.0)
    phi = pd.DataFrame(phi_np, index=CONSTRUCTS, columns=CONSTRUCTS)

    loadings = {i: float(cl.loc[i, i[:2]]) for i in items}
    item_corr = load_item_correlations(item_correlations) if item_correlations is not None else None

    # pesos externos aproximados (modo A: w ∝ λ), na escala de escore com variância 1
    weights = {}
    for c, its in blocks.items():
        lam = np.array([loadings[i] for i in its])
        if item_corr is not None:
            s = item_corr.loc[its, its].to_numpy()
        else:  # matriz implicada pelo modelo de medida
            s = np.outer(lam, lam)
            np.fill_diagonal(s, 1.0)
        weights[c] = [float(v) for v in lam / np.sqrt(lam @ s @ lam)]

    n_obs = int(desc["Number of observations used"].iloc[0]) if "Number of observations used" in desc else None

    # bateria: exata se a matriz de correlação dos itens foi fornecida; senão, implicada por λ
    reliability = {}
    for c, its in blocks.items():
        lam = [loadings[i] for i in its]
        if item_corr is not None:
            block_corr = item_corr.loc[its, its]
        else:
            m = np.outer(lam, lam)
            np.fill_diagonal(m, 1.0)
            block_corr = pd.DataFrame(m, index=its, columns=its)
        reliability[c] = {
            "alpha": cronbach_alpha_std(block_corr),
            "rho_a": rho_a(np.array(weights[c]), block_corr),
            "rho_c": composite_reliability(lam),
            "ave": ave(lam),
        }

    corr_for_htmt = item_corr if item_corr is not None else _implied_item_corr(blocks, loadings, phi)
    htmt = htmt_matrix(corr_for_htmt, blocks)
    battery = {
        "n": n_obs,
        "reliability": reliability,
        "loadings": dict(loadings),
        "htmt": {f"{a}<->{b}": float(htmt.loc[a, b])
                 for i, a in enumerate(CONSTRUCTS) for b in CONSTRUCTS[i + 1:]},
        "htmt_max": float(np.nanmax(htmt.to_numpy())),
        "outer_vif": outer_vif(corr_for_htmt, blocks),
        "inner_vif": inner_vif(phi.loc[EXOGENOUS, EXOGENOUS]),
        "f_squared": _f_squared_analytic(phi, beta, r2),
        "paths": {c: float(beta[c]) for c in EXOGENOUS},
        "r2": float(r2),
        "battery_basis": "item_correlations" if item_corr is not None else "model_implied",
    }

    # consistência interna: R² implicado por β'Φβ vs R² informado
    bvec = np.array([beta[c] for c in EXOGENOUS])
    r2_implied = float(bvec @ phi.loc[EXOGENOUS, EXOGENOUS].to_numpy() @ bvec)

    model = CalibratedModel(
        name=name,
        constructs=list(CONSTRUCTS),
        endogenous=ENDOGENOUS,
        blocks=blocks,
        loadings=loadings,
        weights=weights,
        beta={c: float(beta[c]) for c in EXOGENOUS},
        r2=float(r2),
        phi={a: {b: float(phi.loc[a, b]) for b in CONSTRUCTS} for a in CONSTRUCTS},
        means={i: float(desc.loc[i, "mean"]) for i in items},
        sds={i: float(desc.loc[i, "sd"]) for i in items},
        item_labels=labels,
        item_corr=(None if item_corr is None else
                   {a: {b: float(item_corr.loc[a, b]) for b in items} for a in items}),
        item_freqs=None,  # sem microdado não há distribuição empírica (geração usa média+DP)
        noise=noise or NoiseParams(),
        provenance={
            "source": source,
            "n": n_obs,
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "estimator": "importado do SmartPLS (sem reestimação; β/R² do diagrama de caminhos)",
            "p_values_smartpls": p_values or {},
            "r2_implied_by_beta_phi": r2_implied,
            "r2_consistency_abs_diff": abs(r2_implied - float(r2)),
            "has_raw_wave": False,
        },
    )
    return model, battery


def _implied_item_corr(blocks: dict[str, list[str]], loadings: dict[str, float],
                       phi: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação dos itens implicada pelo modelo: λ_i λ_j Φ_cd (i≠j)."""
    items = [i for c in CONSTRUCTS for i in blocks[c]]
    out = pd.DataFrame(np.eye(len(items)), index=items, columns=items)
    for a in items:
        for b in items:
            if a != b:
                out.loc[a, b] = loadings[a] * loadings[b] * float(phi.loc[a[:2], b[:2]])
    return out


def _f_squared_analytic(phi: pd.DataFrame, beta: dict[str, float], r2_full: float) -> dict[str, float]:
    """f² analítico a partir de Φ: R² de subconjuntos via r_S' Φ_S⁻¹ r_S.

    Usa as correlações latentes exógeno↔TP do Fornell-Larcker; o R² pleno
    informado entra no denominador (definição f² = (R²_full − R²_-c)/(1 − R²_full)).
    """
    r = np.array([float(phi.loc[c, ENDOGENOUS]) for c in EXOGENOUS])
    phi_exo = phi.loc[EXOGENOUS, EXOGENOUS].to_numpy()
    out = {}
    for j, c in enumerate(EXOGENOUS):
        keep = [k for k in range(len(EXOGENOUS)) if k != j]
        sub = phi_exo[np.ix_(keep, keep)]
        r_sub = r[keep]
        r2_wo = float(r_sub @ np.linalg.solve(sub, r_sub))
        out[c] = float((r2_full - r2_wo) / (1 - r2_full))
    return out
