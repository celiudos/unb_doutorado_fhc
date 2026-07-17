"""Leitura dos CSVs de respostas e dos exports do SmartPLS (delimitador ';', BOM tolerado)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .instrument import NAME_TO_CODE, item_code


def _read_semicolon(path_or_buf) -> pd.DataFrame:
    if isinstance(path_or_buf, (str, Path)):
        text = Path(path_or_buf).read_text(encoding="utf-8-sig")
    else:
        if hasattr(path_or_buf, "seek"):
            path_or_buf.seek(0)  # buffers são relidos (ex.: cargas + wording do mesmo export)
        raw = path_or_buf.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    text = text.replace("﻿", "")  # BOMs perdidos no meio do arquivo (ex.: linha SV1 colada)
    return pd.read_csv(io.StringIO(text), sep=";")


def load_responses(path_or_buf) -> pd.DataFrame:
    """CSV bruto de respostas -> DataFrame com colunas renomeadas para códigos de item.

    Colunas sem código de item (ex.: demografia) são preservadas com o nome original.
    """
    df = _read_semicolon(path_or_buf)
    rename = {}
    for col in df.columns:
        code = item_code(str(col))
        if code:
            rename[col] = code
    df = df.rename(columns=rename)
    item_cols = sorted(rename.values())
    if df[item_cols].isna().any().any():
        raise ValueError("CSV de respostas contém valores faltantes nos itens")
    bad = [(c, float(df[c].min()), float(df[c].max()))
           for c in item_cols if df[c].min() < 1 or df[c].max() > 7]
    if bad:
        raise ValueError(f"Itens fora da escala 1-7: {bad}")
    return df


def load_item_labels(path_or_buf) -> dict[str, str]:
    """Extrai wording dos itens de qualquer export com rótulos '[COD] texto' na 1ª coluna."""
    df = _read_semicolon(path_or_buf)
    out = {}
    for label in df.iloc[:, 0].astype(str):
        code = item_code(label)
        if code:
            out[code] = label.split("]", 1)[1].strip()
    return out


def load_descriptives(path: str | Path) -> pd.DataFrame:
    """Export 'Indicator data (original)': média, DP etc. por item, indexado por código."""
    df = _read_semicolon(path)
    df.index = [item_code(str(v)) for v in df.iloc[:, 0]]
    return df.rename(columns={"Mean": "mean", "Standard deviation": "sd"})


def load_reliability(path: str | Path) -> pd.DataFrame:
    """Export de confiabilidade: alpha, rho_a, rho_c, AVE por construto (índice = código)."""
    df = _read_semicolon(path)
    df.index = [NAME_TO_CODE[str(v).strip()] for v in df.iloc[:, 0]]
    df = df.drop(columns=df.columns[0])
    df.columns = ["alpha", "rho_a", "rho_c", "ave"]
    return df


def load_fornell_larcker(path: str | Path) -> pd.DataFrame:
    """Export Fornell-Larcker: fora da diagonal = correlações latentes; diagonal = raiz do AVE."""
    df = _read_semicolon(path)
    df.index = [NAME_TO_CODE[str(v).strip()] for v in df.iloc[:, 0]]
    df = df.drop(columns=df.columns[0])
    df.columns = [NAME_TO_CODE[str(c).strip()] for c in df.columns]
    full = df.combine_first(df.T)  # espelha o triângulo inferior
    return full.astype(float)


def load_htmt(path: str | Path) -> dict[str, float]:
    """Export HTMT em lista: {'SV<->SG': 0.838, ...} com códigos ordenados canonicamente."""
    from .instrument import CONSTRUCTS

    df = _read_semicolon(path)
    out = {}
    for label, value in zip(df.iloc[:, 0], df.iloc[:, 1]):
        a_name, b_name = [s.strip() for s in str(label).split("<->")]
        a, b = NAME_TO_CODE[a_name], NAME_TO_CODE[b_name]
        if CONSTRUCTS.index(a) > CONSTRUCTS.index(b):
            a, b = b, a
        out[f"{a}<->{b}"] = float(value)
    return out


def load_vif(path: str | Path) -> dict[str, float]:
    df = _read_semicolon(path)
    return {item_code(str(k)): float(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1]) if item_code(str(k))}


def load_item_correlations(path: str | Path) -> pd.DataFrame:
    """Export 'Indicator data (correlations)': matriz 32x32 indexada por código de item."""
    df = _read_semicolon(path)
    df.index = [item_code(str(v)) for v in df.iloc[:, 0]]
    df = df.drop(columns=df.columns[0])
    df.columns = [item_code(str(c)) for c in df.columns]
    return df.astype(float)


def load_cross_loadings(path: str | Path) -> pd.DataFrame:
    """Export de cargas cruzadas (itens x construtos), colunas com códigos."""
    df = _read_semicolon(path)
    df.index = [item_code(str(v)) for v in df.iloc[:, 0]]
    df = df.drop(columns=df.columns[0])
    df.columns = [NAME_TO_CODE[str(c).strip()] for c in df.columns]
    return df.astype(float)
