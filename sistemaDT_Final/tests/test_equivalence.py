"""RF8: equivalência real x sintético e bootstrap dos caminhos."""


from twin_core.equivalence import equivalence_report
from twin_core.generate import generate_sample
from twin_core.model import NoiseParams
from twin_core.pls import bootstrap_paths


def test_equivalence_real_vs_synthetic(calibrated, responses):
    """Sintético fiel (modo empírico, sem ruído) deve passar na bateria RF8 contra o real."""
    model, _ = calibrated
    zero = NoiseParams(p_careless=0, p_straight=0, sigma_acq=0)
    synth, _ = generate_sample(model, n=142, seed=42, noise=zero, mode="empirical")
    rep = equivalence_report(responses, synth, model.blocks, n_perm=200, seed=42)
    assert rep["descriptives"]["max_abs_diff_mean"] < 0.6
    assert min(rep["loading_congruence"].values()) > 0.9
    assert rep["micom_step2"]["all_invariant"]
    assert rep["mga"]["all_equivalent_at_5pct"]


def test_equivalence_detects_broken_synthetic(calibrated, responses):
    """Sintético deliberadamente distorcido (ruído extremo) deve ser reprovado."""
    model, _ = calibrated
    broken = NoiseParams(p_careless=0.9, p_straight=0.05, sigma_acq=0)
    synth, _ = generate_sample(model, n=142, seed=42, noise=broken)
    rep = equivalence_report(responses, synth, model.blocks, n_perm=100, seed=42)
    assert not rep["equivalent"]


def test_bootstrap_significance_pattern(calibrated, responses):
    """Bootstrap deve reproduzir o padrão de significância do SmartPLS:
    SF/SG/SM significativos (p<0.05); SV/SK claramente não significativos."""
    model, _ = calibrated
    boot = bootstrap_paths(responses[model.items()], model.blocks, n_boot=500, seed=42)
    paths = boot["paths"]
    for c in ("SF", "SG", "SM"):
        assert paths[c]["p"] < 0.05, c
    for c in ("SV", "SK"):
        assert paths[c]["p"] > 0.30, c
    lo, hi = boot["r2"]["ci95"]
    assert lo < 0.529 < hi
    assert boot["n_boot"] >= 490
