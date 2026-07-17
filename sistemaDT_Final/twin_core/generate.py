"""RF2 — Geração de respondentes sintéticos, carimbados como sintéticos (G1/G2)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .instrument import EXOGENOUS
from .model import CalibratedModel, NoiseParams, ensure_pd_corr
from .noise import apply_human_noise
from .simulate import Intervention, propagate_tp, sample_exo_latents


def generate_sample(
    model: CalibratedModel,
    n: int,
    seed: int,
    interventions: list[Intervention] | None = None,
    noise: NoiseParams | None = None,
    mode: str = "model",
) -> tuple[pd.DataFrame, dict]:
    """Gera N respondentes sintéticos (escala 1-7) + sidecar de proveniência.

    mode="model": latentes MVN(Φ) -> estrutural -> itens via cargas (suporta intervenções).
    mode="empirical": itens MVN da matriz de correlação empírica 32x32 (maior fidelidade
    item a item; não suporta intervenções).
    """
    if mode not in ("model", "empirical"):
        raise ValueError(f"Modo de geração inválido: {mode}")
    if mode == "empirical" and interventions:
        raise ValueError("Intervenções exigem mode='model' (a matriz empírica não tem estrutura causal)")

    rng = np.random.default_rng(seed)
    noise = noise if noise is not None else model.noise
    items = model.items()

    if mode == "model":
        L_exo = sample_exo_latents(model, n, rng, interventions)
        eta_tp = propagate_tp(model, L_exo, rng)
        eta = {c: L_exo[:, j] for j, c in enumerate(EXOGENOUS)}
        eta[model.endogenous] = eta_tp
        z = {}
        for c in model.constructs:
            for it in model.blocks[c]:
                lam = model.loadings[it]
                eps = rng.normal(0, 1, size=n)
                z[it] = lam * eta[c] + np.sqrt(max(0.0, 1 - lam**2)) * eps
        z = pd.DataFrame(z)[items]
    else:
        if not model.item_corr:
            raise ValueError("Modelo não tem matriz de correlação empírica dos itens")
        R = ensure_pd_corr(pd.DataFrame(model.item_corr).loc[items, items]).to_numpy()
        z = pd.DataFrame(
            rng.multivariate_normal(np.zeros(len(items)), R, size=n, method="cholesky"),
            columns=items,
        )

    # z é N(0,1) marginal em cada item; leva à escala 1-7 por cópula gaussiana
    # (quantile matching na distribuição empírica — reproduz as marginais observadas
    # sem o viés de arredondamento/clip do mapeamento linear média+DP)
    if model.item_freqs:
        from scipy.stats import norm

        cont = {}
        for it in items:
            probs = np.array([model.item_freqs[it][str(k)] for k in range(1, 8)])
            cum = np.cumsum(probs) / probs.sum()
            u = norm.cdf(z[it])
            cont[it] = (np.searchsorted(cum, u, side="left") + 1).clip(1, 7).astype(float)
        cont = pd.DataFrame(cont)
    else:  # fallback (modelos importados sem distribuição empírica): SPEC 6.11 clássico
        cont = pd.DataFrame({it: model.means[it] + model.sds[it] * z[it] for it in items})

    cont = apply_human_noise(cont[items], rng, noise)
    flags = cont[["__careless", "__straightline"]]
    data = cont[items].round().clip(1, 7).astype(int)
    data = data.join(flags)
    data["__synthetic"] = True

    sidecar = {
        "schema": "synthetic_sidecar/v1",
        "is_synthetic": True,
        "model_name": model.name,
        "model_hash": model.provenance_hash(),
        "n": n, "seed": seed, "mode": mode,
        "interventions": [vars(iv) for iv in interventions or []],
        "noise": vars(noise),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warning": "DADOS SINTÉTICOS — não representam coleta empírica real (SPEC Seção 13, G1/G2).",
    }
    return data, sidecar
