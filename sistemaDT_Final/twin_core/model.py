"""Estado paramétrico do gêmeo (Seção 4.3): {λ, β, Φ, médias, DP, R²}, versionado e auditável."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .instrument import EXOGENOUS


@dataclass
class NoiseParams:
    """Ruído comportamental humano (RF2)."""
    p_careless: float = 0.07      # respondentes desatentos (respostas aleatórias)
    p_straight: float = 0.05      # straightlining (mesma resposta em tudo)
    sigma_acq: float = 0.35       # aquiescência (viés individual, DP em unidades da escala)


@dataclass
class CalibratedModel:
    """Modelo calibrado do gêmeo — o 'modelo do gêmeo' da SPEC."""
    name: str
    constructs: list[str]
    endogenous: str
    blocks: dict[str, list[str]]                 # construto -> itens
    loadings: dict[str, float]                   # item -> λ
    weights: dict[str, list[float]]              # construto -> pesos externos
    beta: dict[str, float]                       # exógeno -> β
    r2: float
    phi: dict[str, dict[str, float]]             # correlações latentes (todos os construtos)
    means: dict[str, float]                      # item -> média (escala original)
    sds: dict[str, float]                        # item -> DP (escala original)
    item_labels: dict[str, str] = field(default_factory=dict)
    item_corr: dict[str, dict[str, float]] | None = None   # matriz empírica 32x32 (opcional)
    item_freqs: dict[str, dict[str, float]] | None = None  # distribuição empírica 1-7 por item (cópula)
    noise: NoiseParams = field(default_factory=NoiseParams)
    provenance: dict = field(default_factory=dict)

    # ---------- helpers numéricos ----------

    def phi_exo(self) -> pd.DataFrame:
        df = pd.DataFrame(self.phi)
        return df.loc[EXOGENOUS, EXOGENOUS].astype(float)

    def beta_vector(self) -> np.ndarray:
        return np.array([self.beta[c] for c in EXOGENOUS])

    def items(self) -> list[str]:
        return [i for c in self.constructs for i in self.blocks[c]]

    def tp_scale_factor(self) -> float:
        """Conversão ΔTP latente (DP) -> pontos no índice TP (média dos itens TP, escala 1-7)."""
        tp_items = self.blocks[self.endogenous]
        return float(np.mean([self.sds[i] * self.loadings[i] for i in tp_items]))

    # ---------- serialização ----------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = "calibrated_model/v1"
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def provenance_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "CalibratedModel":
        d = dict(d)
        d.pop("schema", None)
        d["noise"] = NoiseParams(**d.get("noise", {}))
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "CalibratedModel":
        return cls.from_dict(json.loads(s))


def _nearest_pd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Projeção simples em matriz positiva definida (clip de autovalores)."""
    sym = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def ensure_pd_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Garante matriz de correlação positiva definida (necessário para Cholesky)."""
    a = df.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(a, 1.0)
    try:
        np.linalg.cholesky(a)
        return pd.DataFrame(a, index=df.index, columns=df.columns)
    except np.linalg.LinAlgError:
        return pd.DataFrame(_nearest_pd(a), index=df.index, columns=df.columns)
