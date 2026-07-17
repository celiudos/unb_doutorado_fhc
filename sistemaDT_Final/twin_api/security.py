"""Autenticação JWT + RBAC do piloto (Seções 10.4 e 12.3).

Papéis: viewer < analyst < admin. Usuários locais via TWIN_USERS
("user:senha:papel,..."); defaults de demonstração se não configurado.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET = os.environ.get("TWIN_SECRET") or secrets.token_hex(32)
ALGO = "HS256"
TOKEN_TTL_S = int(os.environ.get("TWIN_TOKEN_TTL_S", 8 * 3600))

_ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def _load_users() -> dict[str, dict]:
    raw = os.environ.get("TWIN_USERS", "admin:admin123:admin,analista:analista123:analyst,leitor:leitor123:viewer")
    users = {}
    for entry in raw.split(","):
        username, password, role = entry.strip().split(":")
        if role not in _ROLE_RANK:
            raise ValueError(f"Papel desconhecido: {role}")
        salt = secrets.token_hex(8)
        users[username] = {"salt": salt, "hash": _hash(password, salt), "role": role}
    return users


USERS = _load_users()


def authenticate(username: str, password: str) -> dict | None:
    user = USERS.get(username)
    if not user or _hash(password, user["salt"]) != user["hash"]:
        return None
    return {"username": username, "role": user["role"]}


def issue_token(username: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + TOKEN_TTL_S}, SECRET, ALGO)


def current_user(token: str = Depends(oauth2)) -> dict:
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token inválido ou expirado")
    return {"username": payload["sub"], "role": payload["role"]}


def require(role: str):
    def checker(user: dict = Depends(current_user)) -> dict:
        if _ROLE_RANK[user["role"]] < _ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requer papel '{role}'")
        return user

    return checker
