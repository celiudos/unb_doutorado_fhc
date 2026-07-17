"""RF2/RF5/RF6: geração sintética, reprodutibilidade (NF4), what-if e sensibilidade."""

import numpy as np
import pandas as pd
import pytest

from twin_core.generate import generate_sample
from twin_core.model import NoiseParams
from twin_core.sensitivity import ipma, ipma_items, marginal_effects, recommend, tornado
from twin_core.simulate import (
    Intervention,
    baseline_summary,
    monte_carlo_whatif,
    parse_interventions,
    scenario_wargame_phishing,
)


def test_generate_reproducible_and_stamped(calibrated):
    model, _ = calibrated
    a, sc_a = generate_sample(model, n=300, seed=123)
    b, _ = generate_sample(model, n=300, seed=123)
    c, _ = generate_sample(model, n=300, seed=124)
    pd.testing.assert_frame_equal(a, b)                       # NF4: 100% reprodutível
    assert not a[model.items()].equals(c[model.items()])
    assert a["__synthetic"].all()                             # G1: carimbo
    assert sc_a["is_synthetic"] and sc_a["model_hash"] == model.provenance_hash()
    assert a[model.items()].min().min() >= 1 and a[model.items()].max().max() <= 7


def test_generate_recovers_descriptives(calibrated):
    """Sem ruído, o sintético grande deve reproduzir médias/DPs do real (~erro amostral)."""
    model, _ = calibrated
    zero = NoiseParams(p_careless=0, p_straight=0, sigma_acq=0)
    synth, _ = generate_sample(model, n=20000, seed=7, noise=zero)
    for it in model.items():
        assert synth[it].mean() == pytest.approx(model.means[it], abs=0.12), it
        assert synth[it].std(ddof=1) == pytest.approx(model.sds[it], abs=0.15), it


def test_generate_empirical_mode(calibrated):
    model, _ = calibrated
    zero = NoiseParams(p_careless=0, p_straight=0, sigma_acq=0)
    synth, sc = generate_sample(model, n=20000, seed=7, noise=zero, mode="empirical")
    assert sc["mode"] == "empirical"
    corr = synth[model.items()].corr()
    ref = pd.DataFrame(model.item_corr)
    # atenuação por arredondamento/clip é esperada; correlações devem acompanhar de perto
    diffs = [abs(corr.loc[a, b] - ref.loc[a, b]) for a in model.items() for b in model.items() if a != b]
    assert np.mean(diffs) < 0.05

    with pytest.raises(ValueError):
        generate_sample(model, n=10, seed=1, mode="empirical",
                        interventions=[Intervention("SF", "shift", 1.0)])


def test_noise_flags(calibrated):
    model, _ = calibrated
    noisy = NoiseParams(p_careless=0.2, p_straight=0.2, sigma_acq=0.5)
    synth, _ = generate_sample(model, n=5000, seed=11, noise=noisy)
    assert synth["__careless"].mean() == pytest.approx(0.2, abs=0.03)
    assert synth["__straightline"].mean() == pytest.approx(0.2 * 0.8, abs=0.03)
    straight_rows = synth[synth["__straightline"]][model.items()]
    assert (straight_rows.nunique(axis=1) == 1).all()


def test_whatif_shift_matches_beta(calibrated):
    """ΔTP esperado de shift δ em c = escala·β_c·δ (validação analítica do Monte Carlo)."""
    model, _ = calibrated
    eff = marginal_effects(model)
    mc = monte_carlo_whatif(model, [Intervention("SF", "shift", 1.0)], n=400, k=800, seed=5)
    assert mc["delta_tp_latent"]["mean"] == pytest.approx(eff["SF"], abs=0.02)
    lo, hi = mc["delta_tp_latent"]["ci95"]
    assert lo < eff["SF"] < hi
    assert mc["p_delta_positive"] > 0.95
    assert mc["is_synthetic"]


def test_whatif_set_breaks_correlations(calibrated):
    """set-point (do) zera o efeito das correlações: TP médio = escala·β_c·valor,
    demais construtos ficam na média 0 (diferente do shift, que preserva Φ amostrada)."""
    model, _ = calibrated
    from twin_core.simulate import sample_exo_latents
    rng = np.random.default_rng(3)
    L = sample_exo_latents(model, 50000, rng, [Intervention("SG", "set", 2.0)])
    j = ["SV", "SG", "SU", "SM", "SF", "SY", "SK"].index("SG")
    assert L[:, j].std() == pytest.approx(0.0, abs=1e-12)
    others = [k for k in range(7) if k != j]
    assert np.abs(L[:, others].mean(axis=0)).max() < 0.03


def test_wargame_phishing_negative(calibrated):
    model, _ = calibrated
    mc = monte_carlo_whatif(model, scenario_wargame_phishing(), n=300, k=600, seed=9)
    assert mc["delta_tp_latent"]["mean"] < 0
    assert mc["tp_index_projected"] < mc["tp_index_baseline"]


def test_parse_interventions_spec_format():
    ivs = parse_interventions([{"construct": "SK", "tipo": "shift", "delta": -0.6}])
    assert ivs[0].kind == "shift" and ivs[0].value == -0.6
    with pytest.raises(ValueError):
        parse_interventions([{"construct": "TP", "kind": "shift", "delta": 1}])


def test_tornado_ipma_recommend(calibrated):
    model, _ = calibrated
    t = tornado(model)
    assert t[0]["construct"] in ("SF", "SG")             # maiores |β| do estudo
    assert abs(t[0]["delta_up"]) >= abs(t[-1]["delta_up"])

    imp = ipma(model)
    assert all(0 <= r["performance"] <= 100 for r in imp)

    recs = recommend(model, top=3, n=100, k=300, seed=2)
    assert len(recs) == 3
    assert all(r["requires_human_approval"] for r in recs)          # Seção 7: humano decide
    assert all(r["expected_delta_tp_points"] > 0 for r in recs)
    prio = [r["construct"] for r in recs]
    assert "SF" in prio                                   # β=0.376 com desempenho ~64: prioridade


def test_ipma_items(calibrated):
    """IPMA de indicador ('IPMA2'): 28 itens exógenos, coerente com o export do SmartPLS."""
    model, _ = calibrated
    rows = ipma_items(model)
    assert len(rows) == 28                                    # 7 construtos exógenos x 4 itens
    by_item = {r["item"]: r for r in rows}
    # desempenho = média reescalada: SM4 é o item de menor desempenho (~36/100 na imagem SmartPLS)
    assert by_item["SM4"]["performance"] == pytest.approx((model.means["SM4"] - 1) / 6 * 100)
    assert by_item["SM4"]["performance"] < 45
    # importâncias somam o β do construto (pesos normalizados)
    for c in ("SF", "SG", "SM"):
        total = sum(r["importance"] for r in rows if r["construct"] == c)
        assert total == pytest.approx(model.beta[c], abs=1e-9)
    # itens de construtos com β negativo herdam importância negativa (como no SmartPLS)
    assert all(r["importance"] < 0 for r in rows if r["construct"] in ("SU", "SY"))
    # itens de Equipe estão entre os mais importantes (β=0.376)
    top5 = sorted(rows, key=lambda r: r["importance"], reverse=True)[:5]
    assert any(r["construct"] == "SF" for r in top5)


def test_item_levers_in_recommendations(calibrated):
    """2º estágio IPMA: recomendações apontam itens-alavanca com benefício estimado."""
    from twin_core.sensitivity import item_levers

    model, _ = calibrated
    levers = item_levers(model, "SM", top=4)
    assert len(levers) == 4
    # SM4 (recompensas/sanções): pior desempenho da pesquisa -> prioridade no bloco Sistemas
    assert levers[0]["item"] == "SM4"
    assert levers[0]["performance"] < 45
    # benefício coerente: ΔTP por +1 ponto = eff_c · w_unstd · fator_TP, positivo p/ β>0
    assert all(lv["delta_tp_per_point"] > 0 for lv in levers)
    # consistência com o Monte Carlo: shift de 1 DP no construto = soma ponderada dos itens;
    # o benefício de item deve ser fração modesta do efeito do construto inteiro
    eff_points = marginal_effects(model)["SM"] * model.tp_scale_factor()
    assert 0 < levers[0]["delta_tp_per_point"] < eff_points

    recs = recommend(model, top=2, n=100, k=200, seed=3)
    for r in recs:
        assert len(r["levers"]) == 2
        assert all("delta_tp_per_point" in lv and "label" in lv for lv in r["levers"])


def test_recommend_exclude(calibrated):
    """`exclude` retira construtos já em jogo: emissão vira lista de trabalho sem repetição."""
    model, _ = calibrated
    first = recommend(model, top=3, n=80, k=150, seed=1)
    first_c = {r["construct"] for r in first}
    assert len(first_c) == 3

    second = recommend(model, top=3, n=80, k=150, seed=1, exclude=first_c)
    assert first_c.isdisjoint({r["construct"] for r in second})   # nada repete

    # excluindo todos os elegíveis, não sobra nada
    todos = first_c | {r["construct"] for r in second}
    assert recommend(model, top=5, n=80, k=150, seed=1, exclude=todos) == []


def test_baseline_summary(calibrated):
    model, _ = calibrated
    base = baseline_summary(model)
    assert set(base) == set(model.constructs)
    assert 1 <= base["TP"]["mean_1_7"] <= 7


def test_performance_nf1(calibrated):
    """NF1: gerar N=1000 em menos de 1s."""
    import time
    model, _ = calibrated
    start = time.perf_counter()
    generate_sample(model, n=1000, seed=42)
    assert time.perf_counter() - start < 1.0


def test_model_roundtrip_json(calibrated):
    from twin_core.model import CalibratedModel
    model, _ = calibrated
    clone = CalibratedModel.from_json(model.to_json())
    assert clone.beta == model.beta
    assert clone.provenance_hash() == model.provenance_hash()
    a, _ = generate_sample(model, n=50, seed=1)
    b, _ = generate_sample(clone, n=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
