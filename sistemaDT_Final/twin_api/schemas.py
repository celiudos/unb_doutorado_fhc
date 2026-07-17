"""Contratos Pydantic da API (Fase 1)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InterventionIn(BaseModel):
    construct: str
    kind: str = Field(pattern="^(shift|set)$")
    value: float


class NoiseIn(BaseModel):
    p_careless: float = Field(0.07, ge=0, le=1)
    p_straight: float = Field(0.05, ge=0, le=1)
    sigma_acq: float = Field(0.35, ge=0)


class GenerateIn(BaseModel):
    n: int = Field(gt=0, le=100_000)
    seed: int = 42
    mode: str = Field("model", pattern="^(model|empirical)$")
    noise: NoiseIn | None = None
    interventions: list[InterventionIn] = []


class SimulateIn(BaseModel):
    n: int = Field(200, gt=0, le=10_000)
    k: int = Field(2000, gt=0, le=20_000)
    seed: int = 42
    interventions: list[InterventionIn] = []


class ScenarioIn(BaseModel):
    model_id: str
    name: str
    interventions: list[InterventionIn]


class ScenarioRunIn(BaseModel):
    n: int = Field(200, gt=0, le=10_000)
    k: int = Field(2000, gt=0, le=20_000)
    seed: int = 42


class RecommendationDecisionIn(BaseModel):
    status: str = Field(pattern="^(accepted|rejected)$")


class EquivalenceIn(BaseModel):
    dataset_id: str
    n_perm: int = Field(200, gt=0, le=2000)
    seed: int = 42
