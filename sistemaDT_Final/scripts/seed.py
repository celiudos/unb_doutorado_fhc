"""Semeia o gêmeo com a onda 1 real (dados/onda1/respostas_full.csv).

Uso: .venv/bin/python scripts/seed.py [--nome modelo-7s-tp]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twin_core.calibrate import calibrate_from_responses  # noqa: E402
from twin_core.io_smartpls import load_item_labels, load_responses  # noqa: E402
from twin_api.db import VAR_DIR, init_db, session  # noqa: E402
from twin_api.orm import ModelRow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nome", default="modelo-7s-tp-onda1")
    parser.add_argument("--csv", default=str(ROOT / "dados" / "onda1" / "respostas_full.csv"))
    args = parser.parse_args()

    init_db()
    df = load_responses(args.csv)
    labels = load_item_labels(ROOT / "dados" / "onda1" / "cross_loadings.csv")
    model, battery = calibrate_from_responses(df, name=args.nome, item_labels=labels,
                                              source=f"seed:{Path(args.csv).name}")

    with session() as s:
        existing = s.query(ModelRow).filter_by(provenance_hash=model.provenance_hash()).first()
        if existing:
            print(f"Modelo já semeado: {existing.id} ({existing.name})")
            return
        row = ModelRow(
            name=args.nome,
            params_json=model.to_json(),
            battery_json=json.dumps(battery, ensure_ascii=False),
            provenance_hash=model.provenance_hash(),
            wave_csv_path="",
            created_by="seed",
        )
        s.add(row)
        s.flush()
        wave_path = VAR_DIR / "waves" / f"{row.id}_onda1.csv"
        shutil.copy(args.csv, wave_path)
        row.wave_csv_path = str(wave_path)
        s.commit()
        print(f"Modelo calibrado e semeado: {row.id}")
        print(f"  N={battery['n']}  R²={model.r2:.3f}  HTMT máx={battery['htmt_max']:.3f}")
        print("  betas: " + ", ".join(f"{c}={v:+.3f}" for c, v in model.beta.items()))


if __name__ == "__main__":
    main()
