"""Relatório PDF das recomendações aceitas (RF11 — exportação carimbada, G2/G5)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fpdf import FPDF

from twin_core.instrument import CONSTRUCT_NAMES
from twin_core.model import CalibratedModel


_TRANSLIT = str.maketrans({"β": "b", "Δ": "d", "—": "-", "–": "-", "↔": "<->", "≈": "~", "→": "->"})


def _latin(s) -> str:
    """Fontes core do PDF são latin-1; translitera símbolos comuns e substitui o resto."""
    return str(s).translate(_TRANSLIT).encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def __init__(self, provenance_hash: str):
        super().__init__()
        self.provenance_hash = provenance_hash
        self.set_auto_page_break(auto=True, margin=18)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 5, _latin(f"Digital Twin da Proteção Tecnológica · modelo {self.provenance_hash} · "
                               f"página {self.page_no()}/{{nb}}"), align="C")


def build_accepted_recommendations_pdf(
    model: CalibratedModel,
    model_id: str,
    accepted: list[dict],
    generated_by: str,
) -> bytes:
    pdf = _Report(model.provenance_hash())
    pdf.alias_nb_pages()
    pdf.add_page()

    # título
    pdf.set_font("helvetica", "B", 15)
    pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=7, text=_latin("Relatório de Recomendações Aceitas"))
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(90)
    pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=6, text=_latin("Digital Twin Cognitivo-Organizacional da Proteção Tecnológica "
                                "(7S -> TP, PLS-SEM) - SPEC v2.1, Fase 1"))
    pdf.ln(2)

    # metadados do modelo
    pdf.set_text_color(0)
    pdf.set_font("helvetica", "", 10)
    meta = [
        ("Modelo", f"{model.name} ({model_id})"),
        ("Hash de proveniência", model.provenance_hash()),
        ("Fonte de calibração", f"{model.provenance.get('source', '-')} · N={model.provenance.get('n', '-')}"),
        ("R² (Proteção Tecnológica)", f"{model.r2:.3f}"),
        ("Gerado em", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("Gerado por", generated_by),
        ("Recomendações aceitas", str(len(accepted))),
    ]
    for k, v in meta:
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(58, 6, _latin(k))
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=6, text=_latin(v))
    pdf.ln(2)

    # aviso de integridade (G2/G5)
    pdf.set_fill_color(243, 240, 254)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=5, text=_latin(
        "AVISO: os efeitos esperados são projeções Monte Carlo sob o modelo PLS-SEM estimado "
        "(dados de simulação sintéticos, rotulados como tais). O modelo é correlacional; os valores "
        "não constituem inferência causal experimental. Toda recomendação foi decidida por um humano "
        "(SPEC Seção 7 - autoridade graduada, nível 'aprovação humana')."), fill=True)
    pdf.ln(4)

    if not accepted:
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=6, text=_latin("Nenhuma recomendação aceita no momento. Aceite recomendações na aba "
                                    "Prescritivo ou emita novas antes de gerar este relatório."))
    for idx, rec in enumerate(accepted, 1):
        expected = rec["expected"]
        iv = rec["intervention"]
        name = CONSTRUCT_NAMES.get(rec["construct"], rec["construct"])

        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(232, 237, 251)
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=7, text=_latin(f"{idx}. {name} - {iv.get('kind', 'shift')} de {iv.get('value')} DP"),
                       fill=True)
        pdf.set_font("helvetica", "", 10)
        ci = expected.get("ci95_points", [None, None])
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=6, text=_latin(
            f"Efeito esperado: +{expected.get('expected_delta_tp_points', 0):.2f} pontos no índice TP (1-7) · "
            f"IC 95% [{ci[0]:.2f}; {ci[1]:.2f}] · P(aumento)={expected.get('p_positive', 0) * 100:.0f}%"))
        pdf.set_text_color(90)
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=5, text=_latin(f"Justificativa: {expected.get('rationale', '-')}"))
        pdf.set_text_color(0)

        levers = expected.get("levers") or []
        if levers:
            pdf.set_font("helvetica", "B", 10)
            pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=6, text=_latin("Onde atuar (itens-alavanca, IPMA de indicador):"))
            pdf.set_font("helvetica", "", 9)
            for lv in levers:
                pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=5, text=_latin(
                    f"  · [{lv['item']}] {lv['label']} - desempenho {lv['performance']:.0f}/100; "
                    f"+1 ponto neste item = +{lv['delta_tp_per_point']:.2f} em TP"))
        pdf.set_font("helvetica", "I", 9)
        pdf.set_text_color(90)
        pdf.multi_cell(0, new_x="LMARGIN", new_y="NEXT", h=5, text=_latin(f"Aceita por {rec.get('decided_by', '-')} em {rec.get('decided_at', '-')} · "
                                    f"id {rec['id']}"))
        pdf.set_text_color(0)
        pdf.ln(3)

    return bytes(pdf.output())


def rec_row_to_dict(row) -> dict:
    return {
        "id": row.id,
        "construct": row.construct,
        "intervention": json.loads(row.intervention_json),
        "expected": json.loads(row.expected_json),
        "status": row.status,
        "decided_by": row.decided_by,
        "decided_at": row.decided_at,
    }
