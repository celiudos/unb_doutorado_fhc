"""Rotas de modelo: calibração (RF1), bateria (RF3), geração (RF2), sensibilidade (RF6)."""

from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from twin_core.calibrate import calibrate_from_responses, calibrate_from_smartpls
from twin_core.generate import generate_sample
from twin_core.io_smartpls import load_responses
from twin_core.model import CalibratedModel, NoiseParams
from twin_core.pls import bootstrap_paths
from twin_core.sensitivity import ipma, ipma_items, marginal_effects, recommend, tornado
from twin_core.simulate import baseline_summary, parse_interventions

from ..db import VAR_DIR, get_session
from ..orm import AuditLogRow, DatasetRow, ModelRow, RecommendationRow
from ..schemas import GenerateIn, RecommendationDecisionIn
from ..security import require

router = APIRouter(tags=["models"])


def _get_model_row(session: Session, model_id: str) -> ModelRow:
    row = session.get(ModelRow, model_id)
    if not row:
        raise HTTPException(404, "Modelo não encontrado")
    return row


def load_model(session: Session, model_id: str) -> tuple[ModelRow, CalibratedModel]:
    row = _get_model_row(session, model_id)
    return row, CalibratedModel.from_json(row.params_json)


def _audit(session: Session, actor: str, action: str, entity: str, entity_id: str, detail: dict | None = None):
    session.add(AuditLogRow(actor=actor, action=action, entity=entity, entity_id=entity_id,
                            detail=json.dumps(detail or {}, ensure_ascii=False)))


@router.post("/models", status_code=201)
async def create_model(
    file: UploadFile,
    name: str = "modelo-7s-tp",
    session: Session = Depends(get_session),
    user: dict = Depends(require("admin")),
):
    """RF1: calibra o gêmeo a partir do CSV bruto de respostas (onda 1)."""
    content = await file.read()
    try:
        df = load_responses(io.BytesIO(content))
        model, battery = calibrate_from_responses(df, name=name, source=f"upload:{file.filename}")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"Calibração falhou: {exc}")

    row = ModelRow(
        name=name,
        params_json=model.to_json(),
        battery_json=json.dumps(battery, ensure_ascii=False),
        provenance_hash=model.provenance_hash(),
        wave_csv_path="",
        created_by=user["username"],
    )
    session.add(row)
    session.flush()
    wave_path = VAR_DIR / "waves" / f"{row.id}_onda1.csv"
    wave_path.write_bytes(content)
    row.wave_csv_path = str(wave_path)
    _audit(session, user["username"], "calibrate", "model", row.id, {"n": battery["n"]})
    session.commit()
    return model_summary(row, model)


@router.post("/models/smartpls", status_code=201)
async def create_model_from_smartpls(
    cross_loadings: UploadFile,
    fornell_larcker: UploadFile,
    descriptives: UploadFile,
    beta: str = Form(...),
    r2: float = Form(...),
    name: str = Form("modelo-smartpls"),
    p_values: str = Form("{}"),
    item_correlations: UploadFile | None = None,
    session: Session = Depends(get_session),
    user: dict = Depends(require("admin")),
):
    """RF1 (2º caminho): calibra o gêmeo direto dos exports do SmartPLS (sem microdado).

    O what-if passa a rodar sobre o modelo AFERIDO PELO SMARTPLS. β e R² vêm do
    diagrama de caminhos (informados no formulário); bootstrap/equivalência exigem
    o CSV bruto e ficam indisponíveis para modelos importados.
    """
    try:
        beta_dict = {k: float(v) for k, v in json.loads(beta).items()}
        p_dict = {k: float(v) for k, v in json.loads(p_values).items()}
        corr_buf = io.BytesIO(await item_correlations.read()) if item_correlations else None
        model, battery = calibrate_from_smartpls(
            cross_loadings=io.BytesIO(await cross_loadings.read()),
            fornell_larcker=io.BytesIO(await fornell_larcker.read()),
            descriptives=io.BytesIO(await descriptives.read()),
            beta=beta_dict, r2=r2, p_values=p_dict,
            item_correlations=corr_buf, name=name,
        )
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"Importação SmartPLS falhou: {exc}")

    row = ModelRow(
        name=name,
        params_json=model.to_json(),
        battery_json=json.dumps(battery, ensure_ascii=False),
        provenance_hash=model.provenance_hash(),
        wave_csv_path="",  # sem microdado: bootstrap/equivalência indisponíveis
        created_by=user["username"],
    )
    session.add(row)
    session.flush()
    _audit(session, user["username"], "calibrate_smartpls", "model", row.id,
           {"r2": r2, "battery_basis": battery.get("battery_basis")})
    session.commit()
    return model_summary(row, model)


def model_summary(row: ModelRow, model: CalibratedModel) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "provenance_hash": row.provenance_hash,
        "created_by": row.created_by,
        "created_at": row.created_at,
        "n": model.provenance.get("n"),
        "r2": model.r2,
        "beta": model.beta,
        "constructs": model.constructs,
        "maturity_level": "1-3 (descritivo a preditivo, laço bidirecional discreto)",  # CA9
    }


@router.get("/models")
def list_models(session: Session = Depends(get_session), user: dict = Depends(require("viewer"))):
    rows = session.query(ModelRow).order_by(ModelRow.created_at.desc()).all()
    return [model_summary(r, CalibratedModel.from_json(r.params_json)) for r in rows]


@router.get("/models/{model_id}")
def get_model(model_id: str, session: Session = Depends(get_session), user: dict = Depends(require("viewer"))):
    row, model = load_model(session, model_id)
    out = model_summary(row, model)
    out.update({
        "loadings": model.loadings, "phi": model.phi, "means": model.means, "sds": model.sds,
        "item_labels": model.item_labels, "blocks": model.blocks,
        "noise": vars(model.noise), "provenance": model.provenance,
    })
    return out


@router.get("/models/{model_id}/battery")
def get_battery(model_id: str, session: Session = Depends(get_session), user: dict = Depends(require("viewer"))):
    row = _get_model_row(session, model_id)
    return json.loads(row.battery_json)


@router.get("/models/{model_id}/baseline")
def get_baseline(model_id: str, session: Session = Depends(get_session), user: dict = Depends(require("viewer"))):
    _, model = load_model(session, model_id)
    return {"constructs": baseline_summary(model), "r2": model.r2, "is_synthetic_source": False}


@router.get("/models/{model_id}/bootstrap")
def get_bootstrap(
    model_id: str,
    n_boot: int = 500,
    seed: int = 42,
    session: Session = Depends(get_session),
    user: dict = Depends(require("analyst")),
):
    """RF3/RF8: significância dos caminhos por bootstrap (reestimação do bruto da onda 1)."""
    row, model = load_model(session, model_id)
    if not row.wave_csv_path:
        raise HTTPException(409, "Modelo importado de parâmetros SmartPLS não tem onda bruta; "
                                 "bootstrap exige o CSV de respostas (use os p-valores do SmartPLS "
                                 "registrados na proveniência)")
    df = load_responses(row.wave_csv_path)
    return bootstrap_paths(df[model.items()], model.blocks, n_boot=min(n_boot, 2000), seed=seed)


@router.get("/models/{model_id}/sensitivity")
def get_sensitivity(model_id: str, session: Session = Depends(get_session),
                    user: dict = Depends(require("viewer"))):
    """RF6: efeitos marginais, tornado, IPMA de construtos e IPMA de itens."""
    _, model = load_model(session, model_id)
    return {"marginal_effects": marginal_effects(model), "tornado": tornado(model),
            "ipma": ipma(model), "ipma_items": ipma_items(model)}


@router.post("/models/{model_id}/generate", status_code=201)
def generate(
    model_id: str,
    body: GenerateIn,
    session: Session = Depends(get_session),
    user: dict = Depends(require("analyst")),
):
    """RF2: gera dataset sintético carimbado (G1/G2), persistido com sidecar de proveniência."""
    _, model = load_model(session, model_id)
    noise = NoiseParams(**body.noise.model_dump()) if body.noise else None
    try:
        ivs = parse_interventions([iv.model_dump() for iv in body.interventions])
        data, sidecar = generate_sample(model, n=body.n, seed=body.seed,
                                        interventions=ivs, noise=noise, mode=body.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    ds = DatasetRow(model_id=model_id, n=body.n, seed=body.seed, mode=body.mode,
                    path="", sidecar_json=json.dumps(sidecar, ensure_ascii=False),
                    created_by=user["username"])
    session.add(ds)
    session.flush()
    path = VAR_DIR / "datasets" / f"{ds.id}.csv"
    data.to_csv(path, index=False, sep=";")
    ds.path = str(path)
    _audit(session, user["username"], "generate", "dataset", ds.id, {"n": body.n, "seed": body.seed})
    session.commit()
    return {"dataset_id": ds.id, "sidecar": sidecar,
            "preview": data.head(5).to_dict(orient="records")}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, session: Session = Depends(get_session),
                user: dict = Depends(require("viewer"))):
    ds = session.get(DatasetRow, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset não encontrado")
    return {"id": ds.id, "model_id": ds.model_id, "n": ds.n, "seed": ds.seed, "mode": ds.mode,
            "is_synthetic": bool(ds.is_synthetic), "sidecar": json.loads(ds.sidecar_json),
            "created_by": ds.created_by, "created_at": ds.created_at}


@router.get("/datasets/{dataset_id}/csv", response_class=PlainTextResponse)
def download_dataset(dataset_id: str, session: Session = Depends(get_session),
                     user: dict = Depends(require("viewer"))):
    """Export carimbado (G2): aviso de dados sintéticos no cabeçalho do arquivo."""
    ds = session.get(DatasetRow, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset não encontrado")
    body = pd.read_csv(ds.path, sep=";").to_csv(index=False, sep=";")
    stamp = "# DADOS SINTETICOS - nao representam coleta empirica real (SPEC Secao 13)\n"
    return PlainTextResponse(stamp + body, media_type="text/csv")


@router.post("/models/{model_id}/recommendations", status_code=201)
def emit_recommendations(
    model_id: str,
    top: int = 3,
    delta: float = 0.5,
    session: Session = Depends(get_session),
    user: dict = Depends(require("analyst")),
):
    """RF6: emite recomendações priorizadas (nível 'aprovação humana' — Seção 7.1).

    Só emite construtos que ainda NÃO estão em jogo: exclui os que já têm
    recomendação pendente, aceita ou rejeitada. Assim a emissão é uma lista de
    trabalho sem repetição. Construtos arquivados (via 'Zerar aceitas') voltam a
    ser elegíveis. Retorna [] quando não há novidade a sugerir.
    """
    _, model = load_model(session, model_id)
    in_play = {c for (c,) in session.query(RecommendationRow.construct)
               .filter_by(model_id=model_id)
               .filter(RecommendationRow.status.in_(("emitted", "accepted", "rejected"))).all()}
    recs = recommend(model, delta=delta, top=top, n=200, k=1000, seed=42, exclude=in_play)
    out = []
    for r in recs:
        row = RecommendationRow(
            model_id=model_id, construct=r["construct"],
            intervention_json=json.dumps(r["intervention"]),
            expected_json=json.dumps({k: r[k] for k in
                                      ("expected_delta_tp_points", "ci95_points", "p_positive",
                                       "rationale", "levers")},
                                     ensure_ascii=False),
            created_by=user["username"],
        )
        session.add(row)
        session.flush()
        out.append({"id": row.id, **r, "status": "emitted"})
    _audit(session, user["username"], "emit_recommendations", "model", model_id, {"count": len(out)})
    session.commit()
    return out


@router.get("/models/{model_id}/recommendations")
def list_recommendations(model_id: str, session: Session = Depends(get_session),
                         user: dict = Depends(require("viewer"))):
    from ..report import rec_row_to_dict

    rows = (session.query(RecommendationRow).filter_by(model_id=model_id)
            .filter(RecommendationRow.status != "archived")
            .order_by(RecommendationRow.created_at.desc()).all())
    return [rec_row_to_dict(r) for r in rows]


@router.get("/models/{model_id}/recommendations/report.pdf")
def accepted_recommendations_report(model_id: str, session: Session = Depends(get_session),
                                    user: dict = Depends(require("viewer"))):
    """RF11: relatório PDF carimbado, somente com as recomendações aceitas."""
    from fastapi.responses import Response

    from ..report import build_accepted_recommendations_pdf, rec_row_to_dict

    row, model = load_model(session, model_id)
    accepted = (session.query(RecommendationRow)
                .filter_by(model_id=model_id, status="accepted")
                .order_by(RecommendationRow.decided_at.asc()).all())
    pdf = build_accepted_recommendations_pdf(model, model_id,
                                             [rec_row_to_dict(r) for r in accepted],
                                             generated_by=user["username"])
    _audit(session, user["username"], "report_accepted_recs", "model", model_id,
           {"accepted": len(accepted)})
    session.commit()
    filename = f"recomendacoes_aceitas_{row.name}_{row.provenance_hash}.pdf"
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/models/{model_id}/recommendations/clear-accepted")
def clear_accepted_recommendations(model_id: str, session: Session = Depends(get_session),
                                   user: dict = Depends(require("analyst"))):
    """'Zerar aceitas': arquiva (não apaga) para preservar a trilha de auditoria (Seção 12.4)."""
    rows = session.query(RecommendationRow).filter_by(model_id=model_id, status="accepted").all()
    for r in rows:
        r.status = "archived"
    _audit(session, user["username"], "clear_accepted_recs", "model", model_id,
           {"archived": len(rows), "ids": [r.id for r in rows]})
    session.commit()
    return {"archived": len(rows)}


@router.patch("/recommendations/{rec_id}")
def decide_recommendation(
    rec_id: str,
    body: RecommendationDecisionIn,
    session: Session = Depends(get_session),
    user: dict = Depends(require("analyst")),
):
    """V->R: registra a decisão humana sobre a recomendação (CA5)."""
    from datetime import datetime, timezone

    row = session.get(RecommendationRow, rec_id)
    if not row:
        raise HTTPException(404, "Recomendação não encontrada")
    if row.status != "emitted":
        raise HTTPException(409, f"Recomendação já decidida ({row.status})")
    row.status = body.status
    row.decided_by = user["username"]
    row.decided_at = datetime.now(timezone.utc).isoformat()
    _audit(session, user["username"], f"recommendation_{body.status}", "recommendation", rec_id)
    session.commit()
    return {"id": row.id, "status": row.status, "decided_by": row.decided_by, "decided_at": row.decided_at}
