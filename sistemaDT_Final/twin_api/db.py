"""Persistência (perfil piloto: SQLite; DATABASE_URL troca para Postgres sem código novo)."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

VAR_DIR = Path(os.environ.get("TWIN_VAR_DIR", Path(__file__).resolve().parents[1] / "var"))
VAR_DIR.mkdir(parents=True, exist_ok=True)
(VAR_DIR / "datasets").mkdir(exist_ok=True)
(VAR_DIR / "waves").mkdir(exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{VAR_DIR / 'twin.db'}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session():
    with SessionLocal() as session:  # pragma: no cover - wiring
        yield session


def init_db() -> None:
    from . import orm  # noqa: F401  (registra as tabelas)

    Base.metadata.create_all(engine)


def session() -> Session:
    return SessionLocal()
