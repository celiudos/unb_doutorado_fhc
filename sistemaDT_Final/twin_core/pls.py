"""Estimador PLS-SEM leve (modo A, esquema de ponderação de caminhos).

Cobre o modelo da SPEC: exógenos correlacionados -> um endógeno (TP).
Reproduz o SmartPLS dentro de tolerância (testes golden em tests/).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def standardize(a: np.ndarray) -> np.ndarray:
    """Padroniza colunas (média 0, DP amostral 1)."""
    return (a - a.mean(axis=0)) / a.std(axis=0, ddof=1)


@dataclass
class PLSResult:
    constructs: list[str]
    endogenous: str
    blocks: dict[str, list[str]]
    weights: dict[str, np.ndarray]          # pesos externos por construto (escala: score com DP 1)
    scores: pd.DataFrame                    # escores latentes padronizados (n x construtos)
    loadings: dict[str, float]              # carga de cada item no próprio construto
    beta: dict[str, float]                  # coeficientes de caminho exógeno -> endógeno
    r2: float
    phi: pd.DataFrame                       # correlações entre escores (todos os construtos)
    iterations: int = 0
    converged: bool = True
    boot: dict = field(default_factory=dict)  # preenchido por bootstrap_paths


def _inner_outer_iteration(
    X: np.ndarray,
    idx: dict[str, list[int]],
    constructs: list[str],
    endogenous: str,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[dict[str, np.ndarray], int, bool]:
    """Itera pesos externos (modo A) com esquema de ponderação de caminhos."""
    exo = [c for c in constructs if c != endogenous]
    n = X.shape[0]
    w = {c: np.ones(len(idx[c])) / np.sqrt(len(idx[c])) for c in constructs}

    def scores_from(wts: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for c in constructs:
            s = X[:, idx[c]] @ wts[c]
            cols.append(s / s.std(ddof=1))
        return np.column_stack(cols)

    it, converged = 0, False
    for it in range(1, max_iter + 1):
        Y = scores_from(w)
        y_end = Y[:, constructs.index(endogenous)]
        Y_exo = Y[:, [constructs.index(c) for c in exo]]

        # esquema de caminhos: endógeno <- OLS dos preditores; exógenos <- corr com sucessor
        b = np.linalg.lstsq(Y_exo, y_end, rcond=None)[0]
        Z = np.empty_like(Y)
        Z[:, constructs.index(endogenous)] = Y_exo @ b
        for c in exo:
            r = float(np.corrcoef(Y[:, constructs.index(c)], y_end)[0, 1])
            Z[:, constructs.index(c)] = r * y_end

        w_new, delta = {}, 0.0
        for c in constructs:
            z = Z[:, constructs.index(c)]
            wc = X[:, idx[c]].T @ z / n            # modo A: covariância item x proxy interno
            norm = np.linalg.norm(wc)
            if norm == 0:
                raise RuntimeError(f"Pesos degenerados no construto {c}")
            wc = wc / norm
            if wc.sum() < 0:                        # convenção de sinal: bloco majoritariamente positivo
                wc = -wc
            delta = max(delta, float(np.max(np.abs(wc - w[c] / np.linalg.norm(w[c])))))
            w_new[c] = wc
        w = w_new
        if delta < tol:
            converged = True
            break
    return w, it, converged


def pls_sem(df: pd.DataFrame, blocks: dict[str, list[str]], endogenous: str = "TP") -> PLSResult:
    """Estima o modelo PLS (medida reflexiva + estrutural exógenos -> endógeno)."""
    constructs = list(blocks)
    items = [i for c in constructs for i in blocks[c]]
    X = standardize(df[items].to_numpy(dtype=float))
    pos = {i: k for k, i in enumerate(items)}
    idx = {c: [pos[i] for i in blocks[c]] for c in constructs}

    w, iters, converged = _inner_outer_iteration(X, idx, constructs, endogenous)

    # escores finais padronizados
    cols = {}
    for c in constructs:
        s = X[:, idx[c]] @ w[c]
        cols[c] = s / s.std(ddof=1)
    scores = pd.DataFrame(cols, index=df.index)

    # cargas: corr(item, escore do próprio construto)
    loadings = {}
    for c in constructs:
        for i in blocks[c]:
            loadings[i] = float(np.corrcoef(X[:, pos[i]], scores[c])[0, 1])

    # estrutural: OLS dos escores
    exo = [c for c in constructs if c != endogenous]
    Y_exo = scores[exo].to_numpy()
    y_end = scores[endogenous].to_numpy()
    b = np.linalg.lstsq(Y_exo, y_end, rcond=None)[0]
    y_hat = Y_exo @ b
    r2 = float(1 - ((y_end - y_hat) ** 2).sum() / ((y_end - y_end.mean()) ** 2).sum())

    # pesos reescalados para aplicar direto em dados padronizados (score com DP 1)
    weights = {}
    for c in constructs:
        s = X[:, idx[c]] @ w[c]
        weights[c] = w[c] / s.std(ddof=1)

    phi = scores.corr()
    return PLSResult(
        constructs=constructs, endogenous=endogenous, blocks=blocks,
        weights=weights, scores=scores, loadings=loadings,
        beta={c: float(v) for c, v in zip(exo, b)}, r2=r2, phi=phi,
        iterations=iters, converged=converged,
    )


def bootstrap_paths(
    df: pd.DataFrame,
    blocks: dict[str, list[str]],
    endogenous: str = "TP",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap dos coeficientes de caminho (reamostragem de linhas, reestimação completa)."""
    rng = np.random.default_rng(seed)
    exo = [c for c in blocks if c != endogenous]
    n = len(df)
    base = pls_sem(df, blocks, endogenous)
    samples = np.empty((n_boot, len(exo)))
    r2s = np.empty(n_boot)
    kept = 0
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        boot_df = df.iloc[take].reset_index(drop=True)
        try:
            res = pls_sem(boot_df, blocks, endogenous)
        except (RuntimeError, np.linalg.LinAlgError):
            continue  # reamostra degenerada (ex.: item sem variância)
        samples[kept] = [res.beta[c] for c in exo]
        r2s[kept] = res.r2
        kept += 1
    samples, r2s = samples[:kept], r2s[:kept]

    out = {"n_boot": kept, "seed": seed, "paths": {}}
    from scipy import stats

    for j, c in enumerate(exo):
        est = base.beta[c]
        se = float(samples[:, j].std(ddof=1))
        t = est / se if se > 0 else float("inf")
        p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
        lo, hi = np.percentile(samples[:, j], [2.5, 97.5])
        out["paths"][c] = {
            "estimate": est, "boot_mean": float(samples[:, j].mean()), "se": se,
            "t": float(t), "p": p, "ci95": [float(lo), float(hi)],
        }
    out["r2"] = {"estimate": base.r2, "ci95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))]}
    return out
