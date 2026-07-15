import io
import math
import os
import base64

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from .GerarDadosUtils import GerarDadosUtils


class UIUtils:
    """Interface Gradio para geração de personas sintéticas via HDTs."""

    ICON_MASCULINO = "👤"
    ICON_FEMININO = "👩"
    ICON_OUTRO = "🧑"

    DIMENSOES = {
        "Shared Values": ["SV1", "SV2", "SV3", "SV4"],
        "Strategy": ["SG1", "SG2", "SG3", "SG4"],
        "Structure": ["SU1", "SU2", "SU3", "SU4"],
        "Systems": ["SM1", "SM2", "SM3", "SM4"],
        "Staff": ["SF1", "SF2", "SF3", "SF4"],
        "Style": ["SY1", "SY2", "SY3", "SY4"],
        "Skills": ["SK1", "SK2", "SK3", "SK4"],
        "Technology Protection": ["TP1", "TP2", "TP3", "TP4"],
    }

    def __init__(self):
        self.utils = None
        self.df_real = None
        self.personas_df = None
        self.df_respostas = None

    def _icon_sexo(self, sexo):
        if sexo == "Masculino":
            return self.ICON_MASCULINO
        elif sexo == "Feminino":
            return self.ICON_FEMININO
        return self.ICON_OUTRO

    @staticmethod
    def _html_progress_bar(pct):
        color = "#4caf50" if pct >= 100 else "#2196f3"
        return (
            f'<div style="width:100%;background:#e0e0e0;border-radius:6px;overflow:hidden;height:24px;">'
            f'<div style="width:{pct}%;background:{color};height:100%;border-radius:6px;'
            f"transition:width 0.3s ease;display:flex;align-items:center;justify-content:center;"
            f'color:#fff;font-size:12px;font-weight:bold;">{pct}%</div></div>'
        )

    @staticmethod
    def _cor_resposta(valor: int | None) -> str:
        """Retorna cor de fundo interpolada de vermelho (1) a verde (7)."""
        if valor is None:
            return "#eee"
        t = max(0, min(1, (valor - 1) / 6))
        r = int(220 - 140 * t)
        g = int(80 + 140 * t)
        return f"rgb({r},{g},80)"

    def _gerar_html_justificativas(self, resultados):
        if not resultados:
            return "<p>Aguardando geração...</p>"

        questionario = {q["item"]: q for q in GerarDadosUtils.QUESTIONARIO_7S}
        cards = []
        for idx, row in enumerate(resultados):
            sexo = row.get("sexo", "")
            icon = self._icon_sexo(sexo)
            persona_id = row.get("persona_id", "?")
            idade = row.get("idade", "?")
            cargo = row.get("funcao_cargo", "")
            justificativa = row.get("justificativa_curta", "Sem justificativa.")

            # Montar tabela de respostas
            linhas_tabela = ""
            for q in GerarDadosUtils.QUESTIONARIO_7S:
                item = q["item"]
                valor = row.get(item)
                cor = self._cor_resposta(
                    int(valor) if valor is not None and str(valor).isdigit() else None
                )
                val_display = valor if valor is not None else "-"
                linhas_tabela += (
                    f"<tr><td>{q['dimensao']}</td><td>{q['afirmacao']}</td>"
                    f'<td style="background:{cor};color:#fff;text-align:center;font-weight:bold;">{val_display}</td></tr>'
                )

            tabela_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:0.8em;margin-top:6px;">'
                '<thead><tr style="background:#333;color:#fff;">'
                "<th style='padding:4px 6px;'>Dimensão</th>"
                "<th style='padding:4px 6px;'>Afirmação</th>"
                "<th style='padding:4px 6px;width:70px;'>Resposta</th>"
                "</tr></thead><tbody>"
                f"{linhas_tabela}</tbody></table>"
            )

            accordion_id = f"acc_{idx}"
            accordion_html = (
                f'<details style="margin-top:8px;"><summary style="cursor:pointer;font-size:0.85em;'
                f'color:#1976d2;font-weight:bold;">📊 Ver respostas detalhadas</summary>'
                f"{tabela_html}</details>"
            )

            cards.append(
                f'<div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:8px 0;'
                f'background:#fafafa;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="font-size:2em;">{icon}</span>'
                f"<strong>{persona_id}</strong> — {sexo}, {idade} anos"
                f"</div>"
                f'<div style="font-size:0.85em;color:#555;margin-bottom:4px;">{cargo}</div>'
                f'<div style="font-style:italic;">"{justificativa}"</div>'
                f"{accordion_html}"
                f"</div>"
            )
        return "".join(cards)

    def _calcular_estatisticas(self, personas_df, bins_idade_str, bins_tempo_str):
        if personas_df is None or personas_df.empty:
            return pd.DataFrame()

        try:
            bins_idade = [int(x.strip()) for x in bins_idade_str.split(",")]
            labels_idade = [
                f"{bins_idade[i] + 1}-{bins_idade[i + 1]}"
                for i in range(len(bins_idade) - 1)
            ]
        except Exception:
            bins_idade = [17, 30, 40, 60]
            labels_idade = ["18-30", "31-40", "41-60"]

        try:
            bins_tempo = [int(x.strip()) for x in bins_tempo_str.split(",")]
            labels_tempo = [
                f"{bins_tempo[i] + 1}-{bins_tempo[i + 1]}"
                for i in range(len(bins_tempo) - 1)
            ]
        except Exception:
            bins_tempo = [-1, 10, 20, 31]
            labels_tempo = ["0-10", "11-20", "21-31"]

        label_map = {
            "idade": "Idade",
            "sexo": "Sexo",
            "setor_trabalha": "Setor Econômico",
            "porte_org": "Porte Organizacional",
            "funcao_cargo": "Cargo Hierárquico",
            "tempo_atuacao_org_anos": "Tempo de Atuação na Organização",
            "ja_recebeu_treinamento": "Já recebeu treinamento",
        }

        all_cats = {
            "idade": labels_idade,
            "sexo": GerarDadosUtils.SEXO_OPCOES,
            "setor_trabalha": GerarDadosUtils.SETOR_TRABALHA_OPCOES,
            "porte_org": GerarDadosUtils.PORTE_ORG_OPCOES,
            "funcao_cargo": GerarDadosUtils.FUNCAO_CARGOS_OPCOES,
            "tempo_atuacao_org_anos": labels_tempo,
            "ja_recebeu_treinamento": GerarDadosUtils.JA_RECEBEU_TREINAMENTO_OPCOES,
        }

        rows = []
        n_total = len(personas_df)
        for col, label in label_map.items():
            if col not in personas_df.columns:
                continue
            if col == "idade":
                series = pd.Series(
                    pd.cut(
                        personas_df[col].astype(float),
                        bins=bins_idade,
                        labels=labels_idade,
                    )
                )
            elif col == "tempo_atuacao_org_anos":
                series = pd.Series(
                    pd.cut(
                        personas_df[col].astype(float),
                        bins=bins_tempo,
                        labels=labels_tempo,
                    )
                )
            else:
                series = personas_df[col]
            counts = series.value_counts().reindex(all_cats[col], fill_value=0)
            for val, n in counts.items():
                pct = f"{100 * n // n_total}%" if n_total > 0 else "0%"
                rows.append({"Atributo": label, "Categoria": val, "n": n, "%": pct})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Comparativo Sintético x Real (gráficos)
    # ------------------------------------------------------------------

    def _gerar_comparativo(self):
        """Gera os gráficos de comparação entre dados reais e sintéticos."""
        if self.df_real is None or self.df_respostas is None:
            return "<p>Execute a geração primeiro para visualizar o comparativo.</p>"

        ITENS = [q["item"] for q in GerarDadosUtils.QUESTIONARIO_7S]
        df_real = self.df_real
        df_sint = self.df_respostas

        # Garantir colunas numéricas
        for item in ITENS:
            if item in df_real.columns:
                df_real[item] = pd.to_numeric(df_real[item], errors="coerce")
            if item in df_sint.columns:
                df_sint[item] = pd.to_numeric(df_sint[item], errors="coerce")

        LIKERT = range(1, 8)
        images_html = []

        # --- Gráfico 1: Distribuição de frequências Likert por dimensão ---
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
        axes_flat = axes.flatten()

        for ax, (dim, itens) in zip(axes_flat, self.DIMENSOES.items()):
            r_cols = [c for c in itens if c in df_real.columns]
            s_cols = [c for c in itens if c in df_sint.columns]
            if not r_cols or not s_cols:
                ax.set_visible(False)
                continue

            r_vals = np.asarray(df_real[r_cols].values).flatten().astype(float)
            s_vals = np.asarray(df_sint[s_cols].values).flatten().astype(float)

            r_vals = r_vals[~np.isnan(r_vals)]
            s_vals = s_vals[~np.isnan(s_vals)]

            r_freq = [np.mean(r_vals == v) * 100 for v in LIKERT]
            s_freq = [np.mean(s_vals == v) * 100 for v in LIKERT]

            x = np.arange(1, 8)
            ax.bar(
                x - 0.2, r_freq, width=0.4, label="Real", color="steelblue", alpha=0.8
            )
            ax.bar(
                x + 0.2,
                s_freq,
                width=0.4,
                label="Sintético",
                color="darkorange",
                alpha=0.8,
            )
            ax.set_title(dim, fontsize=9)
            ax.set_xlabel("Likert (1-7)", fontsize=8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.legend(fontsize=7)

        fig.suptitle(
            "Distribuição de Frequências Likert: Real vs. Sintético",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()
        images_html.append(self._fig_to_html(fig))
        plt.close(fig)

        # --- Gráfico 2: Médias por dimensão ---
        dims = list(self.DIMENSOES.keys())
        medias_r = []
        medias_s = []
        for itens in self.DIMENSOES.values():
            r_cols = [c for c in itens if c in df_real.columns]
            s_cols = [c for c in itens if c in df_sint.columns]
            r_v = (
                np.asarray(df_real[r_cols].values).flatten().astype(float)
                if r_cols
                else np.array([])
            )
            s_v = (
                np.asarray(df_sint[s_cols].values).flatten().astype(float)
                if s_cols
                else np.array([])
            )
            r_v = r_v[~np.isnan(r_v)]
            s_v = s_v[~np.isnan(s_v)]
            medias_r.append(r_v.mean() if len(r_v) > 0 else 0)
            medias_s.append(s_v.mean() if len(s_v) > 0 else 0)

        x = np.arange(len(dims))
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.bar(x - 0.2, medias_r, 0.4, label="Real", color="steelblue", alpha=0.85)
        ax2.bar(
            x + 0.2, medias_s, 0.4, label="Sintético", color="darkorange", alpha=0.85
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(dims, rotation=25, ha="right", fontsize=9)
        ax2.set_ylabel("Média Likert")
        ax2.set_ylim(1, 7)
        ax2.axhline(4, ls="--", color="gray", lw=0.8, label="Ponto neutro (4)")
        ax2.legend()
        ax2.set_title("Média por Dimensão: Real vs. Sintético")
        plt.tight_layout()
        images_html.append(self._fig_to_html(fig2))
        plt.close(fig2)

        return "".join(images_html)

    @staticmethod
    def _fig_to_html(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;margin:12px 0;" />'

    # ------------------------------------------------------------------
    # Geração incremental
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_opcoes(text: str) -> list[str]:
        return [x.strip() for x in text.split(";") if x.strip()]

    @staticmethod
    def _parse_proporcao(text: str) -> list[int]:
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    def _executar_geracao(
        self,
        seed,
        n_personas,
        n_sessoes,
        llm_provider,
        ollama_host,
        ollama_model,
        openai_api_key,
        openai_model,
        bins_idade_str,
        bins_tempo_str,
        idade_min,
        idade_max,
        tempo_atuacao_min,
        tempo_atuacao_max,
        sexo_opcoes_str,
        sexos_proporcao_str,
        setor_opcoes_str,
        setor_proporcao_str,
        porte_opcoes_str,
        porte_proporcao_str,
        cargo_opcoes_str,
        cargo_proporcao_str,
        treinamento_opcoes_str,
        treinamento_proporcao_str,
    ):
        """Generator que faz yield incremental a cada persona processada."""
        self.utils = GerarDadosUtils(
            seed=int(seed),
            n_personas=int(n_personas),
            n_sessoes_por_persona=int(n_sessoes),
            llm_provider=llm_provider,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            idade_min=int(idade_min),
            idade_max=int(idade_max),
            tempo_atuacao_min=int(tempo_atuacao_min),
            tempo_atuacao_max=int(tempo_atuacao_max),
            sexo_opcoes=self._parse_opcoes(sexo_opcoes_str),
            sexos_proporcao=self._parse_proporcao(sexos_proporcao_str),
            setor_trabalha_opcoes=self._parse_opcoes(setor_opcoes_str),
            setor_trabalha_proporcao=self._parse_proporcao(setor_proporcao_str),
            porte_org_opcoes=self._parse_opcoes(porte_opcoes_str),
            porte_org_proporcao=self._parse_proporcao(porte_proporcao_str),
            funcao_cargos_opcoes=self._parse_opcoes(cargo_opcoes_str),
            funcao_cargos_proporcao=self._parse_proporcao(cargo_proporcao_str),
            ja_recebeu_treinamento_opcoes=self._parse_opcoes(treinamento_opcoes_str),
            ja_recebeu_treinamento_proporcao=self._parse_proporcao(
                treinamento_proporcao_str
            ),
        )

        empty_html_comparativo = "<p>Aguardando conclusão...</p>"
        no_file = gr.update(value=None, visible=False)

        yield (
            "⏳ Formatando dados reais...",
            self._html_progress_bar(5),
            pd.DataFrame(),
            "<p>Aguardando...</p>",
            pd.DataFrame(),
            empty_html_comparativo,
            no_file,
        )

        self.df_real = self.utils.formatar_dataset_real("outputs/questionario_real.csv")
        self.df_real.to_csv("outputs/questionario_real_formatado.csv", index=False)

        yield (
            "⏳ Gerando personas...",
            self._html_progress_bar(10),
            pd.DataFrame(),
            "<p>Aguardando...</p>",
            pd.DataFrame(),
            empty_html_comparativo,
            no_file,
        )

        self.personas_df = self.utils.gerar_personas_df()

        df_estatisticas = self._calcular_estatisticas(
            self.personas_df, bins_idade_str, bins_tempo_str
        )

        yield (
            "⏳ Calculando distribuição-alvo e iniciando geração de respostas...",
            self._html_progress_bar(20),
            pd.DataFrame(),
            "<p>Gerando respostas...</p>",
            df_estatisticas,
            empty_html_comparativo,
            no_file,
        )

        target_probs = self.utils.calcular_target_probs(self.df_real)
        personas = self.personas_df.to_dict(orient="records")

        resultados = []
        total = int(n_sessoes) * int(n_personas)
        count = 0

        for sessao in range(1, int(n_sessoes) + 1):
            for persona in personas:
                try:
                    linha = self.utils.responder_persona(
                        persona, target_probs, sessao=sessao
                    )
                    linha["sessao"] = sessao
                    resultados.append(linha)
                except Exception as e:
                    resultados.append({**persona, "erro": str(e), "sessao": sessao})

                count += 1
                pct = int(20 + 70 * (count / total))
                df_parcial = pd.DataFrame(resultados)
                html_parcial = self._gerar_html_justificativas(resultados)

                yield (
                    f"⏳ Processando persona {count}/{total} (sessão {sessao})...",
                    self._html_progress_bar(pct),
                    df_parcial,
                    html_parcial,
                    df_estatisticas,
                    empty_html_comparativo,
                    no_file,
                )

        # Salvar resultados finais
        self.df_respostas = pd.DataFrame(resultados)

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.personas_df.to_csv(output_dir / "hdt_personas.csv", index=False)
        pd.DataFrame(self.utils.QUESTIONARIO_7S).to_csv(
            output_dir / "questionario_7s.csv", index=False
        )
        self.df_respostas.to_csv(output_dir / "hdt_respostas_7s.csv", index=False)

        itens = [q["item"] for q in self.utils.QUESTIONARIO_7S]
        cols_export = ["persona_id", "sexo", "idade"] + itens
        cols_export = [c for c in cols_export if c in self.df_respostas.columns]
        df_export = self.df_respostas[cols_export].rename(
            columns={"persona_id": "Perfil"}  # type: ignore[arg-type]
        )
        df_export.to_csv(
            f"outputs/questionario_sintetico_{len(self.df_respostas)}.csv", index=False
        )

        # Gerar comparativo
        html_comparativo = self._gerar_comparativo()
        html_final = self._gerar_html_justificativas(resultados)

        # Gerar CSV para download
        csv_path = output_dir / "hdt_personas_respostas.csv"
        self.df_respostas.to_csv(csv_path, index=False)

        yield (
            f"✅ Geração concluída: {len(self.df_respostas)} respostas salvas.",
            self._html_progress_bar(100),
            self.df_respostas,
            html_final,
            df_estatisticas,
            html_comparativo,
            gr.update(value=str(csv_path), visible=True),
        )

    def _carregar_proporcoes_reais(self):
        """Lê o dataset real e retorna proporções para preencher os campos da UI."""
        csv_path = Path("outputs/questionario_real.csv")
        if not csv_path.exists():
            csv_path = Path("inputs/questionario_real.csv")
        if not csv_path.exists():
            raise gr.Error(
                "Arquivo questionario_real.csv não encontrado em outputs/ nem inputs/."
            )

        df = GerarDadosUtils(n_personas=1).formatar_dataset_real(str(csv_path))

        def _proporcoes(col_name: str) -> str:
            s: pd.Series = df[col_name]  # type: ignore[assignment]
            counts = s.dropna().value_counts()
            return ",".join(str(int(v)) for v in counts.values)

        def _opcoes(col_name: str) -> str:
            s: pd.Series = df[col_name]  # type: ignore[assignment]
            counts = s.dropna().value_counts()
            return ";".join(str(v) for v in counts.index)

        idade_s: pd.Series = pd.to_numeric(df["idade"], errors="coerce").dropna()  # type: ignore[assignment]
        tempo_s: pd.Series = pd.to_numeric(
            df["tempo_atuacao_org_anos"], errors="coerce"
        ).dropna()  # type: ignore[assignment]

        return (
            int(idade_s.min()) if len(idade_s) > 0 else 23,  # type: ignore[arg-type]
            int(idade_s.max()) if len(idade_s) > 0 else 59,  # type: ignore[arg-type]
            _opcoes("sexo"),
            _proporcoes("sexo"),
            _opcoes("setor_trabalha"),
            _proporcoes("setor_trabalha"),
            _opcoes("porte_org"),
            _proporcoes("porte_org"),
            _opcoes("funcao_cargo"),
            _proporcoes("funcao_cargo"),
            int(tempo_s.min()) if len(tempo_s) > 0 else 0,  # type: ignore[arg-type]
            int(tempo_s.max()) if len(tempo_s) > 0 else 31,  # type: ignore[arg-type]
            _opcoes("ja_recebeu_treinamento"),
            _proporcoes("ja_recebeu_treinamento"),
        )

    def _atualizar_estatisticas(self, bins_idade_str, bins_tempo_str):
        return self._calcular_estatisticas(
            self.personas_df, bins_idade_str, bins_tempo_str
        )

    def _atualizar_comparativo(self):
        return self._gerar_comparativo()

    # ------------------------------------------------------------------
    # PLS-SEM
    # ------------------------------------------------------------------

    def _gerar_pls_sem(self) -> str:
        """Calcula o modelo PLS-SEM e retorna o diagrama como HTML img."""
        df = self.df_respostas
        if df is None or df.empty:
            if self.df_real is not None and not self.df_real.empty:
                df = self.df_real
            else:
                return "<p>Execute a geração primeiro ou carregue o dataset real.</p>"

        constructs = {
            "SV": ["SV1", "SV2", "SV3", "SV4"],
            "SG": ["SG1", "SG2", "SG3", "SG4"],
            "SU": ["SU1", "SU2", "SU3", "SU4"],
            "SM": ["SM1", "SM2", "SM3", "SM4"],
            "SF": ["SF1", "SF2", "SF3", "SF4"],
            "SY": ["SY1", "SY2", "SY3", "SY4"],
            "SK": ["SK1", "SK2", "SK3", "SK4"],
            "TP": ["TP1", "TP2", "TP3", "TP4"],
        }
        labels = {
            "SV": "Shared Values", "SG": "Strategy", "SU": "Structure",
            "SM": "Systems", "SF": "Staff", "SY": "Style",
            "SK": "Skills", "TP": "Technology Protection",
        }
        hypotheses = {"SV": "H1", "SG": "H2", "SU": "H3", "SM": "H4", "SF": "H5", "SY": "H6", "SK": "H7"}
        predictors = ["SV", "SG", "SU", "SM", "SF", "SY", "SK"]

        # Garantir numérico
        all_items = [i for items in constructs.values() for i in items]
        missing = [c for c in all_items if c not in df.columns]
        if missing:
            return f"<p>Colunas ausentes no dataset: {missing}</p>"
        for col in all_items:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Escores latentes
        scores = pd.DataFrame(
            {c: df[items].mean(axis=1) for c, items in constructs.items()}
        ).dropna()

        k = len(predictors)
        if len(scores) < k + 2:
            return "<p>Dados insuficientes para estimar o modelo PLS-SEM.</p>"

        # Padronizar e estimar betas
        std: pd.Series = scores.std(ddof=0)  # type: ignore[assignment]
        std = std.where(std != 0, np.nan)  # type: ignore[arg-type]
        z = (scores - scores.mean()) / std
        z = z.dropna()
        X = z[predictors].to_numpy(dtype=float)
        y = z["TP"].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - y_hat) ** 2)) / sst if sst > 0 else 0.0
        n = len(y)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

        # Bootstrap p-values
        rng = np.random.default_rng(42)
        cols = predictors + ["TP"]
        matrix = scores[cols].to_numpy(dtype=float)
        n_boot = 5000
        boot_betas_list: list[np.ndarray] = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = matrix[idx]
            s = m.std(axis=0, ddof=0)
            if np.any(s == 0):
                continue
            zb = (m - m.mean(axis=0)) / s
            boot_betas_list.append(
                np.linalg.lstsq(zb[:, :k], zb[:, k], rcond=None)[0]
            )
        if len(boot_betas_list) < 100:
            return "<p>Bootstrap falhou — variância insuficiente nos dados.</p>"
        boot_arr = np.array(boot_betas_list)
        se = boot_arr.std(axis=0, ddof=1)
        se = np.where(se == 0, 1e-10, se)
        t_vals = beta / se
        p_vals = np.array([math.erfc(abs(t) / math.sqrt(2)) for t in t_vals])

        # Gerar gráfico matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-1, 11)
        ax.set_ylim(-0.5, len(predictors) - 0.5)
        ax.axis("off")

        tp_x, tp_y = 9, len(predictors) / 2 - 0.5
        ax.text(
            tp_x, tp_y,
            f"Technology\nProtection\nR²={r2:.3f}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#c8e6c9", ec="#388e3c", lw=1.5),
        )

        for i, c in enumerate(predictors):
            y_pos = len(predictors) - 1 - i
            ax.text(
                1, y_pos, f"{labels[c]}\n({c})",
                ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="#bbdefb", ec="#1976d2", lw=1),
            )
            color = "green" if p_vals[i] < 0.05 and beta[i] > 0 else "gray"
            lw = 2.0 if p_vals[i] < 0.05 else 0.8
            ax.annotate(
                "", xy=(7.5, tp_y), xytext=(2.5, y_pos),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )
            mid_x = 5.0
            mid_y = (y_pos + tp_y) / 2
            sig = "*" if p_vals[i] < 0.05 else ""
            ax.text(
                mid_x, mid_y,
                f"{hypotheses[c]}: β={beta[i]:.3f}{sig}",
                ha="center", va="center", fontsize=7, color=color,
            )

        ax.set_title("PLS-SEM: Modelo 7S → Technology Protection", fontsize=11, pad=12)
        plt.tight_layout()
        html = self._fig_to_html(fig)
        plt.close(fig)
        return html

    def criar_interface(self):
        with gr.Blocks(title="Gerador de Personas Sintéticas - HDT 7S") as demo:
            with gr.Tabs():
                with gr.TabItem("ℹ️ Sobre"):
                    sobre_path = Path(__file__).parent / "Sobre.md"
                    sobre_content = sobre_path.read_text(encoding="utf-8")
                    gr.Markdown(sobre_content)

                with gr.TabItem("📂 Passo 1 - Dataset Real"):
                    gr.Markdown("# 📂 Dataset Real")
                    gr.Markdown(
                        "Gerencie o arquivo de respostas reais usado como base para a geração sintética."
                    )

                    CSV_PATH = Path("outputs/questionario_real.csv")
                    COLUNAS_ESPERADAS = [
                        "idade",
                        "sexo",
                        "setor_trabalha",
                        "porte_org",
                        "funcao_cargo",
                        "tempo_atuacao_org_anos",
                        "ja_recebeu_treinamento",
                        "SV1",
                        "SV2",
                        "SV3",
                        "SV4",
                        "SG1",
                        "SG2",
                        "SG3",
                        "SG4",
                        "SU1",
                        "SU2",
                        "SU3",
                        "SU4",
                        "SM1",
                        "SM2",
                        "SM3",
                        "SM4",
                        "SF1",
                        "SF2",
                        "SF3",
                        "SF4",
                        "SY1",
                        "SY2",
                        "SY3",
                        "SY4",
                        "SK1",
                        "SK2",
                        "SK3",
                        "SK4",
                        "TP1",
                        "TP2",
                        "TP3",
                        "TP4",
                    ]

                    CSV_PATH_INPUT = Path("inputs/questionario_real.csv")

                    # --- Status atual ---
                    if CSV_PATH.exists():
                        status_dataset = gr.Markdown(
                            f"✅ Arquivo encontrado: `{CSV_PATH}`"
                        )
                    elif CSV_PATH_INPUT.exists():
                        status_dataset = gr.Markdown(
                            f"ℹ️ Arquivo inicial encontrado em `{CSV_PATH_INPUT}`. Faça upload para salvar em `{CSV_PATH}`."
                        )
                    else:
                        status_dataset = gr.Markdown(
                            f"⚠️ Arquivo não encontrado. Faça upload ou insira manualmente."
                        )

                    # --- Visualização do dataset atual ---
                    def _carregar_dataset_atual():
                        path = CSV_PATH if CSV_PATH.exists() else CSV_PATH_INPUT
                        if path.exists():
                            try:
                                df = (
                                    self.utils.formatar_dataset_real(str(path))
                                    if self.utils
                                    else GerarDadosUtils(
                                        n_personas=1
                                    ).formatar_dataset_real(str(path))
                                )
                                cols_order = [
                                    c for c in COLUNAS_ESPERADAS if c in df.columns
                                ]
                                return df[cols_order]
                            except Exception:
                                return pd.DataFrame()
                        return pd.DataFrame()

                    tbl_dataset_real = gr.Dataframe(
                        label="Dataset Real Atual",
                        value=_carregar_dataset_atual(),
                        interactive=False,
                    )

                    # --- Exemplo e campos esperados ---
                    with gr.Accordion("📋 Campos esperados e exemplo", open=False):
                        gr.Markdown(
                            "O CSV deve conter as seguintes colunas (separadas por vírgula):\n\n"
                            f"`{', '.join(COLUNAS_ESPERADAS)}`\n\n"
                            "- **SV1–TP4**: respostas Likert de 1 a 7\n"
                            "- **idade**: número inteiro (ex: 35)\n"
                            "- **sexo**: Masculino, Feminino ou Prefiro não informar\n"
                            "- **setor_trabalha**: Pública ou Privada\n"
                            "- **porte_org**: Grande (250 ou mais), Média (50 a 249) ou Pequena (10 a 49)\n"
                            "- **funcao_cargo**: Operacional/técnica (analista, desenvolvedor, técnico), Coordenação/supervisão, Gerência ou Direção/executivo\n"
                            "- **tempo_atuacao_org_anos**: número inteiro (ex: 10)\n"
                            "- **ja_recebeu_treinamento**: Sim ou Não"
                        )
                        gr.Markdown(
                            "**Exemplo de linha CSV:**\n\n"
                            '`6,6,5,4,6,3,3,2,6,7,7,6,7,5,4,2,3,2,3,6,2,4,5,5,7,6,5,4,5,4,3,2,36,Masculino,Pública,Grande (250 ou mais),"Operacional/técnica (analista, desenvolvedor, técnico)",11,Não`'
                        )

                    # --- Upload CSV ---
                    gr.Markdown("### 📤 Upload de CSV")
                    upload_csv = gr.File(
                        label="Selecione o arquivo CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    btn_upload = gr.Button("💾 Salvar CSV enviado", variant="primary")
                    upload_status = gr.Markdown("")

                    def _salvar_upload(filepath):
                        if filepath is None:
                            return (
                                "❌ Nenhum arquivo selecionado.",
                                _carregar_dataset_atual(),
                            )
                        try:
                            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                            import shutil

                            shutil.copy(filepath, CSV_PATH)
                            df = GerarDadosUtils(n_personas=1).formatar_dataset_real(
                                str(CSV_PATH)
                            )
                            return f"✅ Arquivo salvo com {len(df)} registros.", df
                        except Exception as e:
                            return f"❌ Erro ao processar: {e}", pd.DataFrame()

                    btn_upload.click(
                        fn=_salvar_upload,
                        inputs=[upload_csv],
                        outputs=[upload_status, tbl_dataset_real],
                    )

                    # --- Inserção manual ---
                    gr.Markdown("### ✏️ Inserção Manual de Linha")
                    with gr.Row():
                        with gr.Column():
                            manual_likert = gr.Textbox(
                                label="Respostas Likert (SV1 a TP4, 32 valores de 1 a 7 separados por vírgula)",
                                placeholder="6,6,5,4,6,3,3,2,6,7,7,6,7,5,4,2,3,2,3,6,2,4,5,5,7,6,5,4,5,4,3,2",
                            )
                        with gr.Column():
                            manual_idade = gr.Number(
                                label="Idade", value=35, precision=0
                            )
                            manual_sexo = gr.Dropdown(
                                label="Sexo",
                                choices=GerarDadosUtils.SEXO_OPCOES,
                                value="Masculino",
                            )
                            manual_setor = gr.Dropdown(
                                label="Setor",
                                choices=GerarDadosUtils.SETOR_TRABALHA_OPCOES,
                                value="Pública",
                            )
                            manual_porte = gr.Dropdown(
                                label="Porte",
                                choices=GerarDadosUtils.PORTE_ORG_OPCOES,
                                value="Grande (250 ou mais)",
                            )
                            manual_cargo = gr.Dropdown(
                                label="Cargo",
                                choices=GerarDadosUtils.FUNCAO_CARGOS_OPCOES,
                                value="Operacional/técnica (analista, desenvolvedor, técnico)",
                            )
                            manual_tempo = gr.Number(
                                label="Tempo de atuação (anos)", value=5, precision=0
                            )
                            manual_treinamento = gr.Dropdown(
                                label="Já recebeu treinamento?",
                                choices=GerarDadosUtils.JA_RECEBEU_TREINAMENTO_OPCOES,
                                value="Não",
                            )

                    btn_manual = gr.Button("➕ Adicionar Linha", variant="secondary")
                    manual_status = gr.Markdown("")

                    def _adicionar_linha_manual(
                        likert_str, idade, sexo, setor, porte, cargo, tempo, treinamento
                    ):
                        try:
                            valores = [v.strip() for v in likert_str.split(",")]
                            if len(valores) != 32:
                                return (
                                    f"❌ Esperados 32 valores Likert, recebidos {len(valores)}.",
                                    _carregar_dataset_atual(),
                                )
                            for v in valores:
                                if not v.isdigit() or not (1 <= int(v) <= 7):
                                    return (
                                        f"❌ Valor Likert inválido: '{v}'. Use 1 a 7.",
                                        _carregar_dataset_atual(),
                                    )

                            itens_likert = COLUNAS_ESPERADAS[7:]
                            row = {itens_likert[i]: int(valores[i]) for i in range(32)}
                            row["idade"] = int(idade)
                            row["sexo"] = sexo
                            row["setor_trabalha"] = setor
                            row["porte_org"] = porte
                            row["funcao_cargo"] = cargo
                            row["tempo_atuacao_org_anos"] = int(tempo)
                            row["ja_recebeu_treinamento"] = treinamento

                            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                            if CSV_PATH.exists():
                                df_existing = GerarDadosUtils(
                                    n_personas=1
                                ).formatar_dataset_real(str(CSV_PATH))
                            else:
                                df_existing = pd.DataFrame(columns=COLUNAS_ESPERADAS)  # type: ignore[arg-type]

                            df_new = pd.concat(
                                [df_existing, pd.DataFrame([row])], ignore_index=True
                            )
                            df_new.to_csv(CSV_PATH, index=False)
                            return (
                                f"✅ Linha adicionada. Total: {len(df_new)} registros.",
                                df_new,
                            )
                        except Exception as e:
                            return f"❌ Erro: {e}", _carregar_dataset_atual()

                    btn_manual.click(
                        fn=_adicionar_linha_manual,
                        inputs=[
                            manual_likert,
                            manual_idade,
                            manual_sexo,
                            manual_setor,
                            manual_porte,
                            manual_cargo,
                            manual_tempo,
                            manual_treinamento,
                        ],
                        outputs=[manual_status, tbl_dataset_real],
                    )

                with gr.TabItem("🧬 Passo 2 - Gerar Personas Sintéticas"):
                    gr.Markdown(
                        "# 🧬 Gerador de Personas Sintéticas (Human Digital Twins)"
                    )
                    gr.Markdown(
                        "Pipeline para criar personas organizacionais brasileiras e gerar respostas sintéticas em escala Likert para o modelo 7S de Proteção Tecnológica."
                    )

                    # --- Configurações ---
                    with gr.Accordion(
                        "⚙️ Seed, Provedor LLM e Parâmetros de Geração",
                        open=False,
                    ):
                        with gr.Row():
                            with gr.Column():
                                seed = gr.Number(label="Seed", value=84, precision=0)
                                n_personas = gr.Number(
                                    label="Nº de Personas", value=3, precision=0
                                )
                                n_sessoes = gr.Number(
                                    label="Sessões por Persona", value=1, precision=0
                                )

                            with gr.Column():
                                llm_provider = gr.Dropdown(
                                    label="Provedor LLM",
                                    choices=["ollama", "openai"],
                                    value="openai",
                                )
                                ollama_host = gr.Textbox(
                                    label="Ollama Host",
                                    value="http://localhost:11434",
                                    visible=False,
                                )
                                ollama_model = gr.Textbox(
                                    label="Ollama Model",
                                    value="gemma4:e2b",
                                    visible=False,
                                )
                                openai_api_key = gr.Textbox(
                                    label="OpenAI API Key",
                                    value=os.getenv("OPENAI_API_KEY", ""),
                                    type="password",
                                    visible=True,
                                )
                                openai_model = gr.Textbox(
                                    label="OpenAI Model",
                                    value="gpt-5.4-nano-2026-03-17",
                                    visible=True,
                                )

                                def _on_provider_change(provider):
                                    is_ollama = provider == "ollama"
                                    return (
                                        gr.update(visible=is_ollama),
                                        gr.update(visible=is_ollama),
                                        gr.update(visible=not is_ollama),
                                        gr.update(visible=not is_ollama),
                                    )

                                llm_provider.change(
                                    fn=_on_provider_change,
                                    inputs=[llm_provider],
                                    outputs=[
                                        ollama_host,
                                        ollama_model,
                                        openai_api_key,
                                        openai_model,
                                    ],
                                )

                    # --- Configuração de Personas ---
                    with gr.Accordion(
                        "🎭 Configuração de Personas (Atributos Demográficos)",
                        open=False,
                    ):
                        gr.Markdown(
                            "Configure o perfil demográfico das personas sintéticas. "
                            "As **proporções** controlam a probabilidade relativa de cada opção ser sorteada "
                            "(ex: `37,8,1` significa que a 1ª opção é ~37× mais provável que a 3ª)."
                        )
                        btn_carregar_proporcoes = gr.Button(
                            "📊 Carregar proporções do dataset real",
                            variant="secondary",
                        )

                        # -- Perfil Pessoal --
                        gr.Markdown("#### 👤 Perfil Pessoal")
                        with gr.Row():
                            idade_min = gr.Number(
                                label="Idade Mínima",
                                value=23,
                                precision=0,
                                info="Limite inferior de idade das personas geradas",
                            )
                            idade_max = gr.Number(
                                label="Idade Máxima",
                                value=59,
                                precision=0,
                                info="Limite superior de idade das personas geradas",
                            )
                        with gr.Row():
                            sexo_opcoes = gr.Textbox(
                                label="Sexo — Opções",
                                value="Masculino;Feminino;Prefiro não informar",
                                info="Valores possíveis separados por ponto-e-vírgula (;)",
                            )
                            sexos_proporcao = gr.Textbox(
                                label="Sexo — Proporções",
                                value="37,8,1",
                                info="Pesos relativos para cada opção, na mesma ordem (separados por vírgula)",
                            )

                        # -- Contexto Organizacional --
                        gr.Markdown("#### 🏢 Contexto Organizacional")
                        with gr.Row():
                            setor_opcoes = gr.Textbox(
                                label="Setor Econômico — Opções",
                                value="Pública;Privada",
                                info="Setores de atuação separados por ponto-e-vírgula (;)",
                            )
                            setor_proporcao = gr.Textbox(
                                label="Setor Econômico — Proporções",
                                value="30,15",
                                info="Pesos relativos para cada setor",
                            )
                        with gr.Row():
                            porte_opcoes = gr.Textbox(
                                label="Porte da Organização — Opções",
                                value="Grande (250 ou mais);Média (50 a 249);Pequena (10 a 49)",
                                info="Classificações de porte separadas por ponto-e-vírgula (;)",
                            )
                            porte_proporcao = gr.Textbox(
                                label="Porte da Organização — Proporções",
                                value="35,10,1",
                                info="Pesos relativos para cada porte",
                            )

                        # -- Perfil Profissional --
                        gr.Markdown("#### 💼 Perfil Profissional")
                        with gr.Row():
                            cargo_opcoes = gr.Textbox(
                                label="Função/Cargo — Opções",
                                value="Operacional/técnica (analista, desenvolvedor, técnico);Coordenação/supervisão;Gerência;Direção/executivo",
                                info="Níveis hierárquicos separados por ponto-e-vírgula (;)",
                            )
                            cargo_proporcao = gr.Textbox(
                                label="Função/Cargo — Proporções",
                                value="30,10,2,1",
                                info="Pesos relativos para cada nível hierárquico",
                            )
                        with gr.Row():
                            tempo_atuacao_min = gr.Number(
                                label="Tempo de Atuação Mínimo (anos)",
                                value=0,
                                precision=0,
                                info="Menor tempo de experiência na organização",
                            )
                            tempo_atuacao_max = gr.Number(
                                label="Tempo de Atuação Máximo (anos)",
                                value=31,
                                precision=0,
                                info="Maior tempo de experiência na organização",
                            )

                        # -- Capacitação --
                        gr.Markdown("#### 🎓 Capacitação")
                        with gr.Row():
                            treinamento_opcoes = gr.Textbox(
                                label="Já recebeu treinamento em proteção tecnológica? — Opções",
                                value="Sim;Não",
                                info="Respostas possíveis separadas por ponto-e-vírgula (;)",
                            )
                            treinamento_proporcao = gr.Textbox(
                                label="Treinamento — Proporções",
                                value="24,22",
                                info="Pesos relativos para cada resposta",
                            )

                    # --- Estatísticas descritivas: parâmetros ---
                    with gr.Accordion(
                        "📊 Parâmetros de Estatísticas Descritivas",
                        open=False,
                    ):
                        with gr.Row():
                            bins_idade_str = gr.Textbox(
                                label="Bins de Idade (separados por vírgula)",
                                value="17, 30, 40, 60",
                                info="Ex: 17,30,40,60 gera faixas 18-30, 31-40, 41-60",
                            )
                            bins_tempo_str = gr.Textbox(
                                label="Bins de Tempo de Atuação (separados por vírgula)",
                                value="-1, 10, 20, 31",
                                info="Ex: -1,10,20,31 gera faixas 0-10, 11-20, 21-31",
                            )

                    btn = gr.Button("🚀 Gerar Personas e Respostas", variant="primary")
                    status = gr.Textbox(label="Status", interactive=False)
                    progress_bar = gr.HTML(value=UIUtils._html_progress_bar(0))

                    # --- Sub-abas de resultados ---
                    with gr.Tabs():
                        with gr.TabItem("📋 Respostas"):
                            tbl_respostas = gr.Dataframe(
                                label="Respostas Sintéticas", interactive=True
                            )
                            download_file = gr.File(
                                label="⬇️ Download CSV (Personas + Respostas)",
                                visible=False,
                            )

                        with gr.TabItem("💬 Justificativas"):
                            gr.Markdown(
                                "Respostas dos usuários sintéticos com ícone de silhueta por sexo:"
                            )
                            html_justificativas = gr.HTML(
                                value="<p>Execute a geração para visualizar.</p>"
                            )

                        with gr.TabItem("📊 Estatísticas Descritivas"):
                            tbl_estatisticas = gr.Dataframe(
                                label="Estatísticas Descritivas das Personas"
                            )
                            btn_recalc = gr.Button(
                                "🔄 Recalcular Estatísticas", variant="secondary"
                            )

                        with gr.TabItem("📈 Comparativo Sintético x Real"):
                            gr.Markdown(
                                "Gráficos de distribuição Likert e médias por dimensão comparando dados reais e sintéticos:"
                            )
                            html_comparativo = gr.HTML(
                                value="<p>Execute a geração para visualizar o comparativo.</p>"
                            )
                            btn_comparativo = gr.Button(
                                "🔄 Atualizar Comparativo", variant="secondary"
                            )

                        with gr.TabItem("📐 PLS-SEM"):
                            gr.Markdown(
                                "Modelo estrutural PLS-SEM: 7S → Technology Protection"
                            )
                            html_pls_sem = gr.HTML(
                                value="<p>Execute a geração para visualizar o modelo PLS-SEM.</p>"
                            )
                            btn_pls_sem = gr.Button(
                                "🔄 Atualizar PLS-SEM", variant="secondary"
                            )



            # --- Eventos ---
            btn.click(
                fn=self._executar_geracao,
                inputs=[
                    seed,
                    n_personas,
                    n_sessoes,
                    llm_provider,
                    ollama_host,
                    ollama_model,
                    openai_api_key,
                    openai_model,
                    bins_idade_str,
                    bins_tempo_str,
                    idade_min,
                    idade_max,
                    tempo_atuacao_min,
                    tempo_atuacao_max,
                    sexo_opcoes,
                    sexos_proporcao,
                    setor_opcoes,
                    setor_proporcao,
                    porte_opcoes,
                    porte_proporcao,
                    cargo_opcoes,
                    cargo_proporcao,
                    treinamento_opcoes,
                    treinamento_proporcao,
                ],
                outputs=[
                    status,
                    progress_bar,
                    tbl_respostas,
                    html_justificativas,
                    tbl_estatisticas,
                    html_comparativo,
                    download_file,
                ],
            )

            btn_recalc.click(
                fn=self._atualizar_estatisticas,
                inputs=[bins_idade_str, bins_tempo_str],
                outputs=[tbl_estatisticas],
            )

            btn_comparativo.click(
                fn=self._atualizar_comparativo,
                inputs=[],
                outputs=[html_comparativo],
            )

            btn_pls_sem.click(
                fn=self._gerar_pls_sem,
                inputs=[],
                outputs=[html_pls_sem],
            )

            btn_carregar_proporcoes.click(
                fn=self._carregar_proporcoes_reais,
                inputs=[],
                outputs=[
                    idade_min,
                    idade_max,
                    sexo_opcoes,
                    sexos_proporcao,
                    setor_opcoes,
                    setor_proporcao,
                    porte_opcoes,
                    porte_proporcao,
                    cargo_opcoes,
                    cargo_proporcao,
                    tempo_atuacao_min,
                    tempo_atuacao_max,
                    treinamento_opcoes,
                    treinamento_proporcao,
                ],
            )

        return demo

    def launch(self, **kwargs):
        shortcut_js = "<script> document.body.classList.remove('dark');</script>"
        _launch_kwargs = {"head": shortcut_js, **kwargs}

        demo = self.criar_interface()
        demo.launch(**_launch_kwargs)
