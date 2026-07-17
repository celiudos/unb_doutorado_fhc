"""Bateria psicométrica (RF3): alfa, rho_A, CR, AVE, HTMT, VIF, f², cargas cruzadas.

Todas as métricas seguem as definições do SmartPLS 4 (alvos de validação do CA2):
o alfa de Cronbach é o padronizado (baseado em correlações).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pls import PLSResult, standardize


def cronbach_alpha_std(corr_block: pd.DataFrame) -> float:
    """Alfa de Cronbach padronizado: k*r̄ / (1 + (k-1)*r̄)."""
    k = corr_block.shape[0]
    off = corr_block.to_numpy()[np.triu_indices(k, 1)]
    rbar = float(off.mean())
    return k * rbar / (1 + (k - 1) * rbar)


def composite_reliability(loadings: list[float]) -> float:
    """rho_c: (Σλ)² / ((Σλ)² + Σ(1-λ²))."""
    lam = np.asarray(loadings)
    num = lam.sum() ** 2
    return float(num / (num + (1 - lam**2).sum()))


def ave(loadings: list[float]) -> float:
    lam = np.asarray(loadings)
    return float((lam**2).mean())


def rho_a(weights: np.ndarray, corr_block: pd.DataFrame) -> float:
    """rho_A de Dijkstra-Henseler.

    `weights` na escala em que o escore w'x tem variância 1 (a fórmula não é
    invariante à escala dos pesos).
    """
    w = np.asarray(weights, dtype=float)
    s = corr_block.to_numpy()
    w = w / np.sqrt(w @ s @ w)  # garante Var(w'x)=1 na métrica de S
    ww = np.outer(w, w)
    num = w @ (s - np.diag(np.diag(s))) @ w
    den = w @ (ww - np.diag(np.diag(ww))) @ w
    return float((w @ w) ** 2 * num / den)


def htmt_matrix(item_corr: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    """HTMT entre todos os pares de construtos (média dos |r| heterotraço /
    média geométrica das médias dos |r| monotraço)."""
    constructs = list(blocks)

    def mono(c: str) -> float:
        sub = item_corr.loc[blocks[c], blocks[c]].to_numpy()
        k = sub.shape[0]
        return float(np.abs(sub[np.triu_indices(k, 1)]).mean())

    mono_mean = {c: mono(c) for c in constructs}
    out = pd.DataFrame(np.nan, index=constructs, columns=constructs)
    for a_i, a in enumerate(constructs):
        for b in constructs[a_i + 1:]:
            hetero = float(np.abs(item_corr.loc[blocks[a], blocks[b]].to_numpy()).mean())
            val = hetero / np.sqrt(mono_mean[a] * mono_mean[b])
            out.loc[a, b] = out.loc[b, a] = val
    return out


def outer_vif(item_corr: pd.DataFrame, blocks: dict[str, list[str]]) -> dict[str, float]:
    """VIF externo: 1/(1-R²) do item regredido nos demais itens do mesmo bloco."""
    out = {}
    for items in blocks.values():
        sub = item_corr.loc[items, items].to_numpy()
        inv = np.linalg.inv(sub)
        for j, it in enumerate(items):
            out[it] = float(inv[j, j])
    return out


def inner_vif(phi_exo: pd.DataFrame) -> dict[str, float]:
    """VIF interno entre preditores (a partir da matriz de correlação dos exógenos)."""
    inv = np.linalg.inv(phi_exo.to_numpy())
    return {c: float(inv[j, j]) for j, c in enumerate(phi_exo.columns)}


def f_squared(res: PLSResult) -> dict[str, float]:
    """f² por preditor: (R²_full - R²_sem_c) / (1 - R²_full)."""
    exo = [c for c in res.constructs if c != res.endogenous]
    y = res.scores[res.endogenous].to_numpy()
    out = {}
    for c in exo:
        others = [o for o in exo if o != c]
        Xo = res.scores[others].to_numpy()
        b = np.linalg.lstsq(Xo, y, rcond=None)[0]
        resid = y - Xo @ b
        r2_wo = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = float((res.r2 - r2_wo) / (1 - res.r2))
    return out


def cross_loadings(df: pd.DataFrame, res: PLSResult) -> pd.DataFrame:
    """Correlação de cada item com o escore de cada construto."""
    items = [i for c in res.constructs for i in res.blocks[c]]
    X = standardize(df[items].to_numpy(dtype=float))
    out = pd.DataFrame(index=items, columns=res.constructs, dtype=float)
    for j, it in enumerate(items):
        for c in res.constructs:
            out.loc[it, c] = float(np.corrcoef(X[:, j], res.scores[c])[0, 1])
    return out


def full_battery(df: pd.DataFrame, res: PLSResult) -> dict:
    """Bateria completa do RF3 sobre um dataset + resultado PLS."""
    items = [i for c in res.constructs for i in res.blocks[c]]
    item_corr = df[items].corr()

    reliability = {}
    for c in res.constructs:
        its = res.blocks[c]
        lams = [res.loadings[i] for i in its]
        block_corr = item_corr.loc[its, its]
        reliability[c] = {
            "alpha": cronbach_alpha_std(block_corr),
            "rho_a": rho_a(res.weights[c], block_corr),
            "rho_c": composite_reliability(lams),
            "ave": ave(lams),
        }

    exo = [c for c in res.constructs if c != res.endogenous]
    htmt = htmt_matrix(item_corr, res.blocks)
    return {
        "n": int(len(df)),
        "reliability": reliability,
        "loadings": dict(res.loadings),
        "htmt": {f"{a}<->{b}": float(htmt.loc[a, b])
                 for i, a in enumerate(res.constructs) for b in res.constructs[i + 1:]},
        "htmt_max": float(np.nanmax(htmt.to_numpy())),
        "outer_vif": outer_vif(item_corr, res.blocks),
        "inner_vif": inner_vif(res.phi.loc[exo, exo]),
        "f_squared": f_squared(res),
        "paths": dict(res.beta),
        "r2": res.r2,
    }
