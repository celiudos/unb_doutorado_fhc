"""RF1 (2º caminho): calibração direto dos exports do SmartPLS + what-if sobre o modelo aferido."""

import numpy as np
import pytest

from twin_core.calibrate import calibrate_from_smartpls
from twin_core.generate import generate_sample
from twin_core.instrument import CONSTRUCTS, EXOGENOUS
from twin_core.io_smartpls import load_reliability
from twin_core.model import NoiseParams
from twin_core.sensitivity import marginal_effects, tornado
from twin_core.simulate import Intervention, monte_carlo_whatif, scenario_wargame_phishing

SMARTPLS_BETA = {"SV": 0.020, "SG": 0.345, "SU": -0.113, "SM": 0.265,
                 "SF": 0.376, "SY": -0.138, "SK": 0.004}
SMARTPLS_P = {"SV": 0.426, "SG": 0.003, "SU": 0.158, "SM": 0.006,
              "SF": 0.000, "SY": 0.092, "SK": 0.485}
SMARTPLS_R2 = 0.529


@pytest.fixture(scope="module")
def imported(data_dir):
    model, battery = calibrate_from_smartpls(
        cross_loadings=data_dir / "cross_loadings.csv",
        fornell_larcker=data_dir / "fornell_larcker.csv",
        descriptives=data_dir / "indicator_descriptives.csv",
        item_correlations=data_dir / "indicator_correlations.csv",
        beta=SMARTPLS_BETA, r2=SMARTPLS_R2, p_values=SMARTPLS_P,
        name="importado-smartpls",
    )
    return model, battery


def test_model_carries_smartpls_parameters_verbatim(imported):
    """O ponto central: o gêmeo usa os parâmetros do SmartPLS SEM reestimação."""
    model, _ = imported
    assert model.beta == pytest.approx(SMARTPLS_BETA)
    assert model.r2 == SMARTPLS_R2
    assert model.provenance["n"] == 142
    assert model.provenance["has_raw_wave"] is False
    assert model.provenance["p_values_smartpls"]["SF"] == 0.000
    # consistência interna β'Φβ vs R² informado (diferenças grandes indicariam
    # transcrição errada do diagrama)
    assert model.provenance["r2_consistency_abs_diff"] < 0.03


def test_battery_matches_smartpls_reliability(imported, data_dir):
    """Com a matriz de correlações fornecida, a bateria recalculada bate com o export."""
    model, battery = imported
    assert battery["battery_basis"] == "item_correlations"
    ref = load_reliability(data_dir / "reliability.csv")
    for c in CONSTRUCTS:
        assert battery["reliability"][c]["alpha"] == pytest.approx(ref.loc[c, "alpha"], abs=0.002), c
        assert battery["reliability"][c]["rho_c"] == pytest.approx(ref.loc[c, "rho_c"], abs=0.02), c
        assert battery["reliability"][c]["ave"] == pytest.approx(ref.loc[c, "ave"], abs=0.02), c
    assert battery["htmt_max"] == pytest.approx(0.838, abs=0.002)


def test_whatif_runs_on_smartpls_model(imported):
    """What-if baseado no modelo aferido pelo SmartPLS: ΔTP = escala·β·δ."""
    model, _ = imported
    eff = marginal_effects(model)
    # escala ≈ 1 (β'Φβ ≈ R²); efeito marginal segue o β do SmartPLS
    assert eff["SF"] == pytest.approx(0.376, abs=0.03)
    assert eff["SG"] == pytest.approx(0.345, abs=0.03)

    mc = monte_carlo_whatif(model, [Intervention("SF", "shift", 1.0)], n=400, k=800, seed=5)
    assert mc["delta_tp_latent"]["mean"] == pytest.approx(eff["SF"], abs=0.02)
    assert mc["p_delta_positive"] > 0.95

    wargame = monte_carlo_whatif(model, scenario_wargame_phishing(), n=300, k=600, seed=9)
    assert wargame["delta_tp_latent"]["mean"] < 0

    t = tornado(model)
    assert t[0]["construct"] == "SF"          # β=0.376 é o maior efeito do estudo


def test_generation_from_imported_model(imported):
    """Sem microdado a geração usa média+DP (fallback documentado): fidelidade ~erro de clip."""
    model, _ = imported
    zero = NoiseParams(p_careless=0, p_straight=0, sigma_acq=0)
    synth, sidecar = generate_sample(model, n=20000, seed=7, noise=zero)
    assert sidecar["is_synthetic"]
    for it in model.items():
        assert synth[it].mean() == pytest.approx(model.means[it], abs=0.25), it
    # modo empírico funciona porque a matriz 32x32 foi fornecida
    emp, _ = generate_sample(model, n=5000, seed=8, noise=zero, mode="empirical")
    corr = emp[model.items()].corr()
    assert abs(corr.loc["SF1", "SF2"] - model.item_corr["SF1"]["SF2"]) < 0.06


def test_import_without_correlations_uses_model_implied(data_dir):
    model, battery = calibrate_from_smartpls(
        cross_loadings=data_dir / "cross_loadings.csv",
        fornell_larcker=data_dir / "fornell_larcker.csv",
        descriptives=data_dir / "indicator_descriptives.csv",
        beta=SMARTPLS_BETA, r2=SMARTPLS_R2,
    )
    assert battery["battery_basis"] == "model_implied"
    assert model.item_corr is None
    # bateria implicada ainda deve ficar perto da real (modelo de medida forte)
    assert battery["reliability"]["SY"]["alpha"] == pytest.approx(0.936, abs=0.05)
    with pytest.raises(ValueError):
        generate_sample(model, n=10, seed=1, mode="empirical")


def test_import_validation_errors(data_dir):
    with pytest.raises(ValueError, match="exógenos"):
        calibrate_from_smartpls(
            cross_loadings=data_dir / "cross_loadings.csv",
            fornell_larcker=data_dir / "fornell_larcker.csv",
            descriptives=data_dir / "indicator_descriptives.csv",
            beta={"SV": 0.1}, r2=0.5,
        )
    with pytest.raises(ValueError, match="R²"):
        calibrate_from_smartpls(
            cross_loadings=data_dir / "cross_loadings.csv",
            fornell_larcker=data_dir / "fornell_larcker.csv",
            descriptives=data_dir / "indicator_descriptives.csv",
            beta=SMARTPLS_BETA, r2=1.5,
        )


def test_phi_diagonal_fixed(imported):
    """Diagonal do Fornell-Larcker (raiz do AVE) precisa virar 1.0 na Φ do gêmeo."""
    model, _ = imported
    for c in CONSTRUCTS:
        assert model.phi[c][c] == 1.0
    for a in EXOGENOUS:
        vals = [model.phi[a][b] for b in CONSTRUCTS if b != a]
        assert all(np.isfinite(v) and abs(v) < 1 for v in vals)
