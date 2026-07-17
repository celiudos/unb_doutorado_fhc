"""Tabelas do piloto (subconjunto Fase 1 do modelo de dados da SPEC Seção 9)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRow(Base):
    __tablename__ = "model"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    params_json: Mapped[str] = mapped_column(Text)          # CalibratedModel serializado
    battery_json: Mapped[str] = mapped_column(Text)         # bateria RF3 da calibração
    provenance_hash: Mapped[str] = mapped_column(String(16))
    wave_csv_path: Mapped[str] = mapped_column(Text)        # onda 1 (R->V)
    created_by: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[str] = mapped_column(String(40), default=_now)


class DatasetRow(Base):
    __tablename__ = "dataset"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("model.id"))
    n: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(16))
    path: Mapped[str] = mapped_column(Text)
    sidecar_json: Mapped[str] = mapped_column(Text)          # proveniência (G1/G6)
    is_synthetic: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[str] = mapped_column(String(40), default=_now)


class ScenarioRow(Base):
    __tablename__ = "scenario"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("model.id"))
    name: Mapped[str] = mapped_column(String(120))
    interventions_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[str] = mapped_column(String(40), default=_now)


class SimulationRow(Base):
    __tablename__ = "simulation_run"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("model.id"))
    scenario_id: Mapped[str | None] = mapped_column(ForeignKey("scenario.id"), nullable=True)
    params_json: Mapped[str] = mapped_column(Text)
    results_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="done")
    created_by: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[str] = mapped_column(String(40), default=_now)


class RecommendationRow(Base):
    __tablename__ = "recommendation"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("model.id"))
    construct: Mapped[str] = mapped_column(String(8))
    intervention_json: Mapped[str] = mapped_column(Text)
    expected_json: Mapped[str] = mapped_column(Text)         # efeito esperado + IC
    status: Mapped[str] = mapped_column(String(16), default="emitted")  # emitted|accepted|rejected
    decided_by: Mapped[str | None] = mapped_column(String(60), nullable=True)
    decided_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    observed_effect: Mapped[float | None] = mapped_column(Float, nullable=True)  # onda seguinte (V->R)
    created_by: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[str] = mapped_column(String(40), default=_now)


class AuditLogRow(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(60))
    action: Mapped[str] = mapped_column(String(60))
    entity: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text, default="{}")
    at: Mapped[str] = mapped_column(String(40), default=_now)
