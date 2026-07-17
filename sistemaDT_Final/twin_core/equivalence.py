"""RF8 — Validação de equivalência real ↔ sintético (descritivos, congruência, MGA, MICOM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instrument import ENDOGENOUS, EXOGENOUS
from .pls import pls_sem, standardize


def compare_descriptives(real: pd.DataFrame, synth: pd.DataFrame, items: list[str]) -> dict:
    rows = {}
    for it in items:
        rows[it] = {
            "mean_real": float(real[it].mean()), "mean_synth": float(synth[it].mean()),
            "sd_real": float(real[it].std(ddof=1)), "sd_synth": float(synth[it].std(ddof=1)),
        }
    d_mean = max(abs(v["mean_real"] - v["mean_synth"]) for v in rows.values())
    d_sd = max(abs(v["sd_real"] - v["sd_synth"]) for v in rows.values())
    return {"items": rows, "max_abs_diff_mean": d_mean, "max_abs_diff_sd": d_sd}


def loading_congruence(real: pd.DataFrame, synth: pd.DataFrame, blocks: dict[str, list[str]]) -> dict:
    """Coeficiente de congruência de Tucker entre os vetores de carga por construto."""
    res_r = pls_sem(real, blocks, ENDOGENOUS)
    res_s = pls_sem(synth, blocks, ENDOGENOUS)
    out = {}
    for c, items in blocks.items():
        a = np.array([res_r.loadings[i] for i in items])
        b = np.array([res_s.loadings[i] for i in items])
        out[c] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return out


def _paths(df: pd.DataFrame, blocks: dict[str, list[str]]) -> np.ndarray:
    res = pls_sem(df, blocks, ENDOGENOUS)
    return np.array([res.beta[c] for c in EXOGENOUS])


def mga_permutation(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    blocks: dict[str, list[str]],
    n_perm: int = 500,
    seed: int = 42,
) -> dict:
    """MGA por permutação: H0 = caminhos iguais nos dois grupos. p alto = equivalente."""
    items = [i for c in blocks for i in blocks[c]]
    rng = np.random.default_rng(seed)
    obs = np.abs(_paths(real[items], blocks) - _paths(synth[items], blocks))

    pooled = pd.concat([real[items], synth[items]], ignore_index=True)
    n1 = len(real)
    count = np.zeros(len(EXOGENOUS))
    done = 0
    for _ in range(n_perm):
        take = rng.permutation(len(pooled))
        try:
            d = np.abs(_paths(pooled.iloc[take[:n1]], blocks) - _paths(pooled.iloc[take[n1:]], blocks))
        except (RuntimeError, np.linalg.LinAlgError):
            continue
        count += d >= obs
        done += 1
    pvals = (count + 1) / (done + 1)
    return {
        "n_perm": done, "seed": seed,
        "paths": {c: {"abs_diff": float(obs[j]), "p": float(pvals[j])} for j, c in enumerate(EXOGENOUS)},
        "all_equivalent_at_5pct": bool((pvals > 0.05).all()),
    }


def micom_step2(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    blocks: dict[str, list[str]],
    n_perm: int = 500,
    seed: int = 42,
) -> dict:
    """MICOM passo 2 (invariância composicional): c = corr(escores com pesos do grupo 1,
    escores com pesos do grupo 2) sobre os dados agrupados; teste por permutação."""
    items = [i for c in blocks for i in blocks[c]]
    rng = np.random.default_rng(seed)
    pooled = pd.concat([real[items], synth[items]], ignore_index=True)
    Xp = standardize(pooled.to_numpy(dtype=float))
    pos = {i: k for k, i in enumerate(items)}

    def c_stats(g1: pd.DataFrame, g2: pd.DataFrame) -> np.ndarray:
        w1 = pls_sem(g1, blocks, ENDOGENOUS).weights
        w2 = pls_sem(g2, blocks, ENDOGENOUS).weights
        cs = []
        for c in blocks:
            cols = [pos[i] for i in blocks[c]]
            s1 = Xp[:, cols] @ np.asarray(w1[c])
            s2 = Xp[:, cols] @ np.asarray(w2[c])
            cs.append(float(np.corrcoef(s1, s2)[0, 1]))
        return np.array(cs)

    obs = c_stats(real[items], synth[items])
    n1 = len(real)
    perm = []
    for _ in range(n_perm):
        take = rng.permutation(len(pooled))
        try:
            perm.append(c_stats(pooled.iloc[take[:n1]], pooled.iloc[take[n1:]]))
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    perm = np.array(perm)
    q5 = np.percentile(perm, 5, axis=0)
    return {
        "n_perm": len(perm), "seed": seed,
        "constructs": {
            c: {"c_value": float(obs[j]), "quantile_5pct": float(q5[j]), "invariant": bool(obs[j] >= q5[j])}
            for j, c in enumerate(blocks)
        },
        "all_invariant": bool((obs >= q5).all()),
    }


def equivalence_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    blocks: dict[str, list[str]],
    n_perm: int = 500,
    seed: int = 42,
) -> dict:
    """Relatório RF8 completo: descritivos, congruência de cargas, MGA e MICOM."""
    items = [i for c in blocks for i in blocks[c]]
    real, synth = real[items].astype(float), synth[items].astype(float)
    desc = compare_descriptives(real, synth, items)
    cong = loading_congruence(real, synth, blocks)
    mga = mga_permutation(real, synth, blocks, n_perm=n_perm, seed=seed)
    micom = micom_step2(real, synth, blocks, n_perm=n_perm, seed=seed)
    verdict = (
        desc["max_abs_diff_mean"] < 0.5
        and min(cong.values()) > 0.95
        and mga["all_equivalent_at_5pct"]
        and micom["all_invariant"]
    )
    return {
        "descriptives": desc,
        "loading_congruence": cong,
        "mga": mga,
        "micom_step2": micom,
        "equivalent": bool(verdict),
        "criteria": {
            "max_abs_diff_mean_lt": 0.5,
            "tucker_congruence_gt": 0.95,
            "mga_all_p_gt": 0.05,
            "micom_c_above_5pct_quantile": True,
        },
    }
