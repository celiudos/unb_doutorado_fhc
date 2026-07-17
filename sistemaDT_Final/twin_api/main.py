"""Aplicação FastAPI do Digital Twin (SPEC v2.1, Fase 1)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routes import models as models_routes
from .routes import simulate as simulate_routes
from .security import authenticate, current_user, issue_token

WEB_DIR = Path(__file__).resolve().parents[1] / "twin_web"

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Digital Twin Cognitivo-Organizacional da Proteção Tecnológica",
    version="0.1.0",
    description="Motor PLS-SEM (7S -> TP) + camada de gêmeo digital. Fase 1 — SPEC v2.1.",
    lifespan=lifespan,
)

auth_router = APIRouter(tags=["auth"])


@auth_router.post("/auth/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if not user:
        raise HTTPException(401, "Usuário ou senha inválidos")
    return {"access_token": issue_token(user["username"], user["role"]), "token_type": "bearer",
            "role": user["role"]}


@auth_router.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.get("/health")
def health():
    return {"status": "ok", "synthetic_data_notice":
            "Todo dado gerado por este sistema é sintético (SPEC Seção 13)."}


app.include_router(auth_router)
app.include_router(models_routes.router)
app.include_router(simulate_routes.router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
