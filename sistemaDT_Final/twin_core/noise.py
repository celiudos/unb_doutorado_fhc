"""Ruído comportamental humano (RF2): desatentos, straightlining, aquiescência."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import NoiseParams


def apply_human_noise(df: pd.DataFrame, rng: np.random.Generator, params: NoiseParams) -> pd.DataFrame:
    """Aplica ruído em valores contínuos (antes de arredondar/clipar).

    Devolve cópia; marca as linhas afetadas nas colunas __careless/__straightline.
    """
    out = df.copy()
    n = len(out)
    items = list(out.columns)

    careless = rng.random(n) < params.p_careless
    straight = (~careless) & (rng.random(n) < params.p_straight)

    # desatentos: respostas uniformes 1..7 em todos os itens
    n_car = int(careless.sum())
    if n_car:
        out.loc[careless, items] = rng.integers(1, 8, size=(n_car, len(items))).astype(float)

    # straightlining: um único valor (levemente enviesado ao centro-alto, como no real)
    n_str = int(straight.sum())
    if n_str:
        vals = rng.choice([3, 4, 5, 6, 7], size=n_str, p=[0.1, 0.2, 0.3, 0.25, 0.15]).astype(float)
        out.loc[straight, items] = np.repeat(vals[:, None], len(items), axis=1)

    # aquiescência: viés individual somado a todos os itens (efeito teto tratado pelo clip)
    bias = rng.normal(0, params.sigma_acq, size=n)
    normal_rows = ~(careless | straight)
    out.loc[normal_rows, items] = out.loc[normal_rows, items].add(bias[normal_rows], axis=0)

    out["__careless"] = careless
    out["__straightline"] = straight
    return out
