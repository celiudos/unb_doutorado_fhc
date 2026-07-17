"""CA2 — o motor deve reproduzir os números do SmartPLS dentro de tolerância.

Alvos: exports reais do estudo (dados/onda1). Tolerâncias declaradas por métrica:
- alfa/HTMT/VIF/correlações: livres de estimação -> 0.002 (arredondamento de exibição)
- cargas/CR/AVE/rho_A/beta/R²: dependem do algoritmo PLS -> 0.02
"""

import numpy as np
import pytest

from twin_core.instrument import CONSTRUCTS, EXOGENOUS
from twin_core.io_smartpls import (
    load_cross_loadings,
    load_fornell_larcker,
    load_htmt,
    load_reliability,
    load_vif,
)

TOL_EXACT = 0.002
TOL_PLS = 0.02

# Betas e p-valores do diagrama SmartPLS (Apêndice A da SPEC v2.1)
SMARTPLS_BETA = {"SV": 0.020, "SG": 0.345, "SU": -0.113, "SM": 0.265,
                 "SF": 0.376, "SY": -0.138, "SK": 0.004}
SMARTPLS_R2 = 0.529


def test_alpha_exact(calibrated, data_dir):
    _, battery = calibrated
    ref = load_reliability(data_dir / "reliability.csv")
    for c in CONSTRUCTS:
        assert battery["reliability"][c]["alpha"] == pytest.approx(ref.loc[c, "alpha"], abs=TOL_EXACT)


def test_rho_c_ave_rho_a(calibrated, data_dir):
    _, battery = calibrated
    ref = load_reliability(data_dir / "reliability.csv")
    for c in CONSTRUCTS:
        assert battery["reliability"][c]["rho_c"] == pytest.approx(ref.loc[c, "rho_c"], abs=TOL_PLS)
        assert battery["reliability"][c]["ave"] == pytest.approx(ref.loc[c, "ave"], abs=TOL_PLS)
        assert battery["reliability"][c]["rho_a"] == pytest.approx(ref.loc[c, "rho_a"], abs=TOL_PLS)


def test_loadings_match_cross_loadings_diagonal(calibrated, data_dir):
    model, _ = calibrated
    ref = load_cross_loadings(data_dir / "cross_loadings.csv")
    for c in CONSTRUCTS:
        for it in model.blocks[c]:
            assert model.loadings[it] == pytest.approx(ref.loc[it, c], abs=TOL_PLS), it


def test_htmt_exact(calibrated, data_dir):
    _, battery = calibrated
    ref = load_htmt(data_dir / "htmt.csv")
    for pair, val in ref.items():
        assert battery["htmt"][pair] == pytest.approx(val, abs=TOL_EXACT), pair
    assert battery["htmt_max"] < 0.85  # validade discriminante do estudo


def test_outer_vif_exact(calibrated, data_dir):
    _, battery = calibrated
    ref = load_vif(data_dir / "vif.csv")
    for it, val in ref.items():
        assert battery["outer_vif"][it] == pytest.approx(val, abs=TOL_EXACT), it


def test_phi_matches_fornell_larcker(calibrated, data_dir):
    model, _ = calibrated
    ref = load_fornell_larcker(data_dir / "fornell_larcker.csv")
    for i, a in enumerate(CONSTRUCTS):
        for b in CONSTRUCTS[i + 1:]:
            assert model.phi[a][b] == pytest.approx(ref.loc[a, b], abs=TOL_PLS), f"{a}<->{b}"


def test_paths_and_r2_match_smartpls(calibrated):
    model, _ = calibrated
    for c in EXOGENOUS:
        assert model.beta[c] == pytest.approx(SMARTPLS_BETA[c], abs=TOL_PLS), c
    assert model.r2 == pytest.approx(SMARTPLS_R2, abs=TOL_PLS)


def test_calibration_provenance(calibrated):
    model, _ = calibrated
    assert model.provenance["n"] == 142
    assert len(model.items()) == 32
    assert all(len(model.blocks[c]) == 4 for c in CONSTRUCTS)
    assert len(model.provenance_hash()) == 16
    assert np.isfinite(model.tp_scale_factor())
