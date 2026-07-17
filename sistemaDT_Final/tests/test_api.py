"""Testes da API: fluxo completo da Fase 1 + RBAC."""

import io

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory, data_dir):
    import os

    var = tmp_path_factory.mktemp("var")
    os.environ["TWIN_VAR_DIR"] = str(var)
    os.environ["DATABASE_URL"] = f"sqlite:///{var / 'test.db'}"

    # módulos leem env na importação: garantir estado limpo
    import importlib
    import twin_api.db, twin_api.orm, twin_api.main  # noqa
    importlib.reload(twin_api.db)
    import twin_api.orm as orm_mod
    importlib.reload(orm_mod)
    import twin_api.routes.models as rm
    importlib.reload(rm)
    import twin_api.routes.simulate as rs
    importlib.reload(rs)
    import twin_api.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as c:
        yield c


def _token(client, username, password):
    r = client.post("/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def admin(client):
    return _token(client, "admin", "admin123")


@pytest.fixture(scope="module")
def analyst(client):
    return _token(client, "analista", "analista123")


@pytest.fixture(scope="module")
def viewer(client):
    return _token(client, "leitor", "leitor123")


@pytest.fixture(scope="module")
def model_id(client, admin, data_dir):
    csv = (data_dir / "respostas_full.csv").read_bytes()
    r = client.post("/models", params={"name": "onda1-api"},
                    files={"file": ("onda1.csv", io.BytesIO(csv), "text/csv")}, headers=admin)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["n"] == 142 and abs(body["r2"] - 0.529) < 0.02
    return body["id"]


@pytest.fixture
def new_model(client, admin, data_dir):
    """Cria um modelo novo por chamada — estado de recomendações limpo, sem interferência entre testes."""
    csv = (data_dir / "respostas_full.csv").read_bytes()

    def _make(name="modelo-teste"):
        r = client.post("/models", params={"name": name},
                        files={"file": ("onda1.csv", io.BytesIO(csv), "text/csv")}, headers=admin)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    return _make


def test_auth_denied(client):
    assert client.get("/models").status_code == 401
    assert client.post("/auth/token", data={"username": "admin", "password": "errada"}).status_code == 401


def test_rbac(client, viewer, analyst, model_id):
    # viewer lê, mas não gera nem simula
    assert client.get(f"/models/{model_id}", headers=viewer).status_code == 200
    r = client.post(f"/models/{model_id}/generate", json={"n": 10, "seed": 1}, headers=viewer)
    assert r.status_code == 403
    # analyst não calibra
    r = client.post("/models", files={"file": ("x.csv", io.BytesIO(b"a;b"), "text/csv")}, headers=analyst)
    assert r.status_code == 403


def test_battery_and_baseline(client, viewer, model_id):
    battery = client.get(f"/models/{model_id}/battery", headers=viewer).json()
    assert battery["htmt_max"] < 0.85
    assert battery["reliability"]["TP"]["alpha"] == pytest.approx(0.854, abs=0.002)
    base = client.get(f"/models/{model_id}/baseline", headers=viewer).json()
    assert 1 <= base["constructs"]["TP"]["mean_1_7"] <= 7


def test_generate_and_download(client, analyst, viewer, model_id):
    r = client.post(f"/models/{model_id}/generate",
                    json={"n": 142, "seed": 42, "mode": "empirical",
                          "noise": {"p_careless": 0, "p_straight": 0, "sigma_acq": 0}},
                    headers=analyst)
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["sidecar"]["is_synthetic"] is True
    csv_text = client.get(f"/datasets/{ds['dataset_id']}/csv", headers=viewer).text
    assert csv_text.startswith("# DADOS SINTETICOS")  # G2

    globals()["_dataset_id"] = ds["dataset_id"]


def test_simulate_whatif(client, analyst, model_id):
    r = client.post(f"/models/{model_id}/simulate",
                    json={"n": 200, "k": 400, "seed": 7,
                          "interventions": [{"construct": "SF", "kind": "shift", "value": 1.0}]},
                    headers=analyst)
    assert r.status_code == 201, r.text
    res = r.json()["results"]
    assert res["delta_tp_latent"]["mean"] > 0.2
    assert res["is_synthetic"] is True

    sim = client.get(f"/simulations/{r.json()['simulation_id']}", headers=analyst)
    assert sim.status_code == 200


def test_scenario_wargame(client, analyst, model_id):
    r = client.post("/scenarios", headers=analyst,
                    json={"model_id": model_id, "name": "Phishing (choque)",
                          "interventions": [{"construct": "SK", "kind": "shift", "value": -0.6},
                                            {"construct": "SF", "kind": "shift", "value": -0.4}]})
    assert r.status_code == 201, r.text
    run = client.post(f"/scenarios/{r.json()['id']}/run", json={"n": 200, "k": 300, "seed": 3},
                      headers=analyst)
    assert run.status_code == 201
    assert run.json()["results"]["delta_tp_latent"]["mean"] < 0


def test_sensitivity_and_recommendations(client, analyst, viewer, model_id, new_model):
    sens = client.get(f"/models/{model_id}/sensitivity", headers=viewer).json()
    assert sens["tornado"][0]["construct"] in ("SF", "SG")

    mid = new_model("recs-decisao")
    recs = client.post(f"/models/{mid}/recommendations", params={"top": 2},
                       headers=analyst).json()
    assert len(recs) == 2 and all(r["status"] == "emitted" for r in recs)

    rec_id = recs[0]["id"]
    dec = client.patch(f"/recommendations/{rec_id}", json={"status": "accepted"}, headers=analyst)
    assert dec.status_code == 200 and dec.json()["status"] == "accepted"
    # segunda decisão sobre a mesma recomendação é rejeitada
    again = client.patch(f"/recommendations/{rec_id}", json={"status": "rejected"}, headers=analyst)
    assert again.status_code == 409


def test_equivalence_endpoint(client, analyst, model_id):
    r = client.post(f"/models/{model_id}/equivalence",
                    json={"dataset_id": globals()["_dataset_id"], "n_perm": 50, "seed": 1},
                    headers=analyst)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert "mga" in rep and "micom_step2" in rep
    assert rep["micom_step2"]["all_invariant"]


def test_smartpls_import_endpoint(client, admin, analyst, data_dir):
    files = {
        "cross_loadings": ("cl.csv", (data_dir / "cross_loadings.csv").read_bytes(), "text/csv"),
        "fornell_larcker": ("fl.csv", (data_dir / "fornell_larcker.csv").read_bytes(), "text/csv"),
        "descriptives": ("d.csv", (data_dir / "indicator_descriptives.csv").read_bytes(), "text/csv"),
        "item_correlations": ("c.csv", (data_dir / "indicator_correlations.csv").read_bytes(), "text/csv"),
    }
    data = {
        "beta": '{"SV":0.020,"SG":0.345,"SU":-0.113,"SM":0.265,"SF":0.376,"SY":-0.138,"SK":0.004}',
        "r2": "0.529",
        "name": "importado-api",
        "p_values": '{"SF":0.000,"SG":0.003}',
    }
    # analyst não pode importar
    assert client.post("/models/smartpls", files=files, data=data, headers=analyst).status_code == 403

    r = client.post("/models/smartpls", files=files, data=data, headers=admin)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["r2"] == 0.529 and body["beta"]["SF"] == 0.376
    mid = body["id"]

    # what-if roda sobre o modelo aferido pelo SmartPLS
    sim = client.post(f"/models/{mid}/simulate",
                      json={"n": 200, "k": 300, "seed": 7,
                            "interventions": [{"construct": "SF", "kind": "shift", "value": 1.0}]},
                      headers=analyst)
    assert sim.status_code == 201, sim.text
    assert sim.json()["results"]["delta_tp_latent"]["mean"] > 0.25

    # sem microdado: bootstrap e equivalência ficam bloqueados com mensagem clara
    assert client.get(f"/models/{mid}/bootstrap", headers=analyst).status_code == 409
    gen = client.post(f"/models/{mid}/generate", json={"n": 50, "seed": 1}, headers=analyst)
    assert gen.status_code == 201
    eq = client.post(f"/models/{mid}/equivalence",
                     json={"dataset_id": gen.json()["dataset_id"], "n_perm": 20, "seed": 1},
                     headers=analyst)
    assert eq.status_code == 409

    # beta incompleto é rejeitado
    bad = dict(data, beta='{"SV":0.1}')
    assert client.post("/models/smartpls", files=files, data=bad, headers=admin).status_code == 422


def test_emit_only_new_constructs(client, analyst, viewer, new_model):
    """Emitir só traz construtos ainda não em jogo (pendente/aceito/rejeitado); sem duplicar."""
    mid = new_model("recs-sem-dup")
    first = client.post(f"/models/{mid}/recommendations", params={"top": 3}, headers=analyst).json()
    first_c = {r["construct"] for r in first}
    assert len(first) == 3

    # segunda emissão não repete nenhum construto já pendente
    second = client.post(f"/models/{mid}/recommendations", params={"top": 3}, headers=analyst).json()
    assert first_c.isdisjoint({r["construct"] for r in second})

    # entre todas as recomendações ativas não há construto repetido
    active = [r for r in client.get(f"/models/{mid}/recommendations", headers=viewer).json()
              if r["status"] in ("emitted", "accepted", "rejected")]
    constructs = [r["construct"] for r in active]
    assert len(constructs) == len(set(constructs))

    # esgotados os elegíveis, emitir retorna vazio (nada novo a sugerir)
    while client.post(f"/models/{mid}/recommendations", params={"top": 3}, headers=analyst).json():
        pass
    assert client.post(f"/models/{mid}/recommendations", params={"top": 3}, headers=analyst).json() == []


def test_report_and_clear_accepted(client, analyst, viewer, new_model):
    """PDF só com aceitas; 'zerar' arquiva e o relatório volta a ficar vazio."""
    mid = new_model("recs-relatorio")
    recs = client.post(f"/models/{mid}/recommendations", params={"top": 2},
                       headers=analyst).json()
    client.patch(f"/recommendations/{recs[0]['id']}", json={"status": "accepted"}, headers=analyst)
    client.patch(f"/recommendations/{recs[1]['id']}", json={"status": "rejected"}, headers=analyst)

    r = client.get(f"/models/{mid}/recommendations/report.pdf", headers=viewer)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert len(r.content) > 1500  # tem conteúdo, não só o esqueleto

    # zerar: viewer não pode; analyst arquiva só a aceita deste modelo
    assert client.post(f"/models/{mid}/recommendations/clear-accepted",
                       headers=viewer).status_code == 403
    out = client.post(f"/models/{mid}/recommendations/clear-accepted", headers=analyst).json()
    assert out["archived"] == 1

    listed = client.get(f"/models/{mid}/recommendations", headers=viewer).json()
    statuses = {x["id"]: x["status"] for x in listed}
    assert recs[0]["id"] not in statuses          # arquivada some da lista
    assert statuses.get(recs[1]["id"]) == "rejected"  # rejeitada permanece visível

    empty = client.get(f"/models/{mid}/recommendations/report.pdf", headers=viewer)
    assert empty.status_code == 200 and empty.content.startswith(b"%PDF")
    assert len(empty.content) < len(r.content)    # relatório vazio é menor


def test_invalid_inputs(client, admin, analyst, model_id):
    bad_csv = b"col1;col2\n1;2"
    r = client.post("/models", files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
                    headers=admin)
    assert r.status_code == 422

    r = client.post(f"/models/{model_id}/simulate",
                    json={"n": 100, "k": 100, "seed": 1,
                          "interventions": [{"construct": "TP", "kind": "shift", "value": 1}]},
                    headers=analyst)
    assert r.status_code == 422
