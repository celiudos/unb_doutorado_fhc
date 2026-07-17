"""Rotas de simulação (RF5), cenários/war-gaming e equivalência (RF8)."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from twin_core.equivalence import equivalence_report
from twin_core.io_smartpls import load_responses
from twin_core.simulate import monte_carlo_whatif, parse_interventions

from ..db import get_session
from ..orm import DatasetRow, ScenarioRow, SimulationRow
from ..schemas import EquivalenceIn, ScenarioIn, ScenarioRunIn, SimulateIn
from ..security import require
from .models import load_model

router = APIRouter(tags=["simulate"])


def _run_mc(session: Session, model_id: str, body_dict: dict, user: dict,
            scenario_id: str | None = None) -> dict:
    _, model = load_model(session, model_id)
    try:
        ivs = parse_interventions(body_dict.get("interventions", []))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    results = monte_carlo_whatif(model, ivs, n=body_dict["n"], k=body_dict["k"], seed=body_dict["seed"])
    run = SimulationRow(
        model_id=model_id, scenario_id=scenario_id,
        params_json=json.dumps(body_dict, ensure_ascii=False),
        results_json=json.dumps(results, ensure_ascii=False),
        created_by=user["username"],
    )
    session.add(run)
    session.commit()
    return {"simulation_id": run.id, "results": results}


@router.post("/models/{model_id}/simulate", status_code=201)
def simulate(model_id: str, body: SimulateIn, session: Session = Depends(get_session),
             user: dict = Depends(require("analyst"))):
    """RF5: Monte Carlo baseline ou what-if direto (sem cenário persistido)."""
    return _run_mc(session, model_id, body.model_dump(), user)


@router.post("/scenarios", status_code=201)
def create_scenario(body: ScenarioIn, session: Session = Depends(get_session),
                    user: dict = Depends(require("analyst"))):
    load_model(session, body.model_id)  # valida existência
    if not body.interventions:
        raise HTTPException(422, "Cenário precisa de ao menos uma intervenção")
    row = ScenarioRow(model_id=body.model_id, name=body.name,
                      interventions_json=json.dumps([iv.model_dump() for iv in body.interventions]),
                      created_by=user["username"])
    session.add(row)
    session.commit()
    return {"id": row.id, "name": row.name, "model_id": row.model_id,
            "interventions": json.loads(row.interventions_json)}


@router.get("/scenarios")
def list_scenarios(model_id: str | None = None, session: Session = Depends(get_session),
                   user: dict = Depends(require("viewer"))):
    q = session.query(ScenarioRow)
    if model_id:
        q = q.filter_by(model_id=model_id)
    return [{"id": r.id, "name": r.name, "model_id": r.model_id,
             "interventions": json.loads(r.interventions_json), "created_at": r.created_at}
            for r in q.order_by(ScenarioRow.created_at.desc()).all()]


@router.post("/scenarios/{scenario_id}/run", status_code=201)
def run_scenario(scenario_id: str, body: ScenarioRunIn, session: Session = Depends(get_session),
                 user: dict = Depends(require("analyst"))):
    sc = session.get(ScenarioRow, scenario_id)
    if not sc:
        raise HTTPException(404, "Cenário não encontrado")
    params = body.model_dump() | {"interventions": json.loads(sc.interventions_json)}
    return _run_mc(session, sc.model_id, params, user, scenario_id=scenario_id)


@router.get("/simulations/{simulation_id}")
def get_simulation(simulation_id: str, session: Session = Depends(get_session),
                   user: dict = Depends(require("viewer"))):
    run = session.get(SimulationRow, simulation_id)
    if not run:
        raise HTTPException(404, "Simulação não encontrada")
    return {"id": run.id, "model_id": run.model_id, "scenario_id": run.scenario_id,
            "params": json.loads(run.params_json), "results": json.loads(run.results_json),
            "status": run.status, "created_at": run.created_at}


@router.post("/models/{model_id}/equivalence")
def validate_equivalence(model_id: str, body: EquivalenceIn, session: Session = Depends(get_session),
                         user: dict = Depends(require("analyst"))):
    """RF8: bateria de equivalência real (onda 1) x sintético (dataset gerado)."""
    row, model = load_model(session, model_id)
    if not row.wave_csv_path:
        raise HTTPException(409, "Modelo importado de parâmetros SmartPLS não tem onda bruta; "
                                 "a equivalência real x sintético exige o CSV de respostas")
    ds = session.get(DatasetRow, body.dataset_id)
    if not ds or ds.model_id != model_id:
        raise HTTPException(404, "Dataset não encontrado para este modelo")
    real = load_responses(row.wave_csv_path)
    synth = pd.read_csv(ds.path, sep=";")
    return equivalence_report(real, synth, model.blocks, n_perm=body.n_perm, seed=body.seed)
