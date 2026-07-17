"""Definições do instrumento 7S -> Proteção Tecnológica (estudo N=142)."""

from __future__ import annotations

import re

# Ordem canônica dos construtos (exógenos + endógeno no fim)
CONSTRUCTS = ["SV", "SG", "SU", "SM", "SF", "SY", "SK", "TP"]
ENDOGENOUS = "TP"
EXOGENOUS = [c for c in CONSTRUCTS if c != ENDOGENOUS]

CONSTRUCT_NAMES = {
    "SV": "Valores Compartilhados",
    "SG": "Estratégia",
    "SU": "Estrutura",
    "SM": "Sistemas",
    "SF": "Equipe",
    "SY": "Estilo de Liderança",
    "SK": "Habilidades",
    "TP": "Proteção Tecnológica",
}
NAME_TO_CODE = {v: k for k, v in CONSTRUCT_NAMES.items()}

_ITEM_RE = re.compile(r"\[([A-Z]{2}\d)\]")


def item_code(label: str) -> str | None:
    """Extrai o código do item (ex.: 'SV1') de um rótulo tipo '[SV1] Minha empresa...'."""
    m = _ITEM_RE.search(label)
    return m.group(1) if m else None


def blocks_from_items(items: list[str]) -> dict[str, list[str]]:
    """Agrupa códigos de item por construto (prefixo de 2 letras), na ordem canônica."""
    out: dict[str, list[str]] = {c: [] for c in CONSTRUCTS}
    for it in items:
        prefix = it[:2]
        if prefix not in out:
            raise ValueError(f"Item '{it}' não pertence a nenhum construto conhecido")
        out[prefix].append(it)
    for c, its in out.items():
        if not its:
            raise ValueError(f"Construto '{c}' sem itens no dataset")
        its.sort()
    return out
