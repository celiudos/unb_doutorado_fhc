import json
import os
import random
import re
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from ollama import Client
from openai import OpenAI
from tqdm.notebook import tqdm


class GerarDadosUtils:
    """Encapsula toda a lógica de geração de dados sintéticos via HDTs."""

    # Atributos demográficos (proporções baseadas nos dados reais)
    IDADE_MIN = 23
    IDADE_MAX = 59
    SEXO_OPCOES = ["Masculino", "Feminino", "Prefiro não informar"]
    SEXOS_PROPORCAO = [37, 8, 1]
    SETOR_TRABALHA_OPCOES = ["Pública", "Privada"]
    SETOR_TRABALHA_PROPORCAO = [30, 15]
    PORTE_ORG_OPCOES = ["Grande (250 ou mais)", "Média (50 a 249)", "Pequena (10 a 49)"]
    PORTE_ORG_PROPORCAO = [35, 10, 1]
    FUNCAO_CARGOS_OPCOES = [
        "Operacional/técnica (analista, desenvolvedor, técnico)",
        "Coordenação/supervisão",
        "Gerência",
        "Direção/executivo",
    ]
    FUNCAO_CARGOS_PROPORCAO = [30, 10, 2, 1]
    TEMPO_ATUACAO_MIN = 0
    TEMPO_ATUACAO_MAX = 31
    JA_RECEBEU_TREINAMENTO_OPCOES = ["Sim", "Não"]
    JA_RECEBEU_TREINAMENTO_PROPORCAO = [24, 22]

    QUESTIONARIO_7S = [
        {
            "item": "SV1",
            "dimensao": "Shared Values",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa compartilha a importância e o valor da proteção tecnológica.",
        },
        {
            "item": "SV2",
            "dimensao": "Shared Values",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa inclui o conceito de proteção tecnológica em suas metas de gestão.",
        },
        {
            "item": "SV3",
            "dimensao": "Shared Values",
            "tipo_variavel": "independente",
            "afirmacao": "Minha organização está ciente da necessidade de proteção tecnológica.",
        },
        {
            "item": "SV4",
            "dimensao": "Shared Values",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa tem uma cultura organizacional que protege sua tecnologia.",
        },
        {
            "item": "SG1",
            "dimensao": "Strategy",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui uma estratégia de proteção tecnológica.",
        },
        {
            "item": "SG2",
            "dimensao": "Strategy",
            "tipo_variavel": "independente",
            "afirmacao": "Para minha empresa, a proteção tecnológica é uma prioridade estratégica.",
        },
        {
            "item": "SG3",
            "dimensao": "Strategy",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui um plano de proteção tecnológica.",
        },
        {
            "item": "SG4",
            "dimensao": "Strategy",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa investe um orçamento suficiente para proteger sua tecnologia.",
        },
        {
            "item": "SU1",
            "dimensao": "Structure",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui uma estrutura organizacional sistemática para proteger sua tecnologia.",
        },
        {
            "item": "SU2",
            "dimensao": "Structure",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui um departamento encarregado de suas responsabilidades de proteção tecnológica.",
        },
        {
            "item": "SU3",
            "dimensao": "Structure",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui um gerente encarregado das responsabilidades de proteção tecnológica.",
        },
        {
            "item": "SU4",
            "dimensao": "Structure",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa opera e gerencia suas responsabilidades de proteção tecnológica de forma eficiente.",
        },
        {
            "item": "SM1",
            "dimensao": "Systems",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui os regulamentos relevantes para proteger sua tecnologia.",
        },
        {
            "item": "SM2",
            "dimensao": "Systems",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa controla e monitora seus dados técnicos.",
        },
        {
            "item": "SM3",
            "dimensao": "Systems",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa usa acordos de confidencialidade e não concorrência para proteger sua tecnologia.",
        },
        {
            "item": "SM4",
            "dimensao": "Systems",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa usa recompensas para funcionários que são excelentes em proteger a tecnologia da empresa e punições para aqueles que violam os regulamentos.",
        },
        {
            "item": "SF1",
            "dimensao": "Staff",
            "tipo_variavel": "independente",
            "afirmacao": "Os membros da minha empresa entendem seus papéis e responsabilidades na proteção da tecnologia da empresa.",
        },
        {
            "item": "SF2",
            "dimensao": "Staff",
            "tipo_variavel": "independente",
            "afirmacao": "Os membros da minha empresa aderem estritamente aos regulamentos e diretrizes relevantes para proteger a tecnologia da empresa.",
        },
        {
            "item": "SF3",
            "dimensao": "Staff",
            "tipo_variavel": "independente",
            "afirmacao": "Os membros da minha empresa recebem treinamento relacionado à proteção tecnológica.",
        },
        {
            "item": "SF4",
            "dimensao": "Staff",
            "tipo_variavel": "independente",
            "afirmacao": "Os membros da minha empresa respeitam os gerentes encarregados da proteção tecnológica.",
        },
        {
            "item": "SY1",
            "dimensao": "Style",
            "tipo_variavel": "independente",
            "afirmacao": "A equipe de gestão da minha empresa tem um alto nível de interesse em proteção tecnológica.",
        },
        {
            "item": "SY2",
            "dimensao": "Style",
            "tipo_variavel": "independente",
            "afirmacao": "A equipe de gestão da minha empresa reconhece que a proteção tecnológica é importante para as atividades de negócios.",
        },
        {
            "item": "SY3",
            "dimensao": "Style",
            "tipo_variavel": "independente",
            "afirmacao": "A equipe de gestão da minha empresa apoia ativamente a proteção da tecnologia da empresa.",
        },
        {
            "item": "SY4",
            "dimensao": "Style",
            "tipo_variavel": "independente",
            "afirmacao": "A equipe de gestão da minha empresa leva a questão da proteção tecnológica em consideração durante os processos de tomada de decisão.",
        },
        {
            "item": "SK1",
            "dimensao": "Skills",
            "tipo_variavel": "independente",
            "afirmacao": "Na minha empresa, a divisão encarregada da proteção da tecnologia tem expertise em proteção tecnológica.",
        },
        {
            "item": "SK2",
            "dimensao": "Skills",
            "tipo_variavel": "independente",
            "afirmacao": "Na minha empresa, os membros do pessoal encarregado da proteção tecnológica têm certificações profissionais.",
        },
        {
            "item": "SK3",
            "dimensao": "Skills",
            "tipo_variavel": "independente",
            "afirmacao": "Na minha empresa, a divisão ou o pessoal encarregado da proteção tecnológica recebe o treinamento profissional relevante.",
        },
        {
            "item": "SK4",
            "dimensao": "Skills",
            "tipo_variavel": "independente",
            "afirmacao": "Minha empresa possui um manual de resposta para a ocorrência de vazamentos de tecnologia.",
        },
        {
            "item": "TP1",
            "dimensao": "Technology Protection",
            "tipo_variavel": "dependente",
            "afirmacao": "Minha empresa protege bem a confidencialidade relacionada à tecnologia.",
        },
        {
            "item": "TP2",
            "dimensao": "Technology Protection",
            "tipo_variavel": "dependente",
            "afirmacao": "Minha empresa é melhor no gerenciamento de segurança do que outras empresas.",
        },
        {
            "item": "TP3",
            "dimensao": "Technology Protection",
            "tipo_variavel": "dependente",
            "afirmacao": "Minha empresa tem menos vazamentos de tecnologia ou incidentes de segurança em comparação com outras empresas.",
        },
        {
            "item": "TP4",
            "dimensao": "Technology Protection",
            "tipo_variavel": "dependente",
            "afirmacao": "Minha empresa nunca sofreu grandes danos por vazamentos de tecnologia ou incidentes de segurança.",
        },
    ]

    def __init__(
        self,
        seed=84,
        n_personas=3,
        n_sessoes_por_persona=1,
        llm_provider="ollama",
        ollama_host="http://localhost:11434",
        ollama_model="gemma4:e2b",
        openai_api_key="",
        openai_model="gpt-5.4-nano-2026-03-17",
        idade_min: int | None = None,
        idade_max: int | None = None,
        sexo_opcoes: list[str] | None = None,
        sexos_proporcao: list[int] | None = None,
        setor_trabalha_opcoes: list[str] | None = None,
        setor_trabalha_proporcao: list[int] | None = None,
        porte_org_opcoes: list[str] | None = None,
        porte_org_proporcao: list[int] | None = None,
        funcao_cargos_opcoes: list[str] | None = None,
        funcao_cargos_proporcao: list[int] | None = None,
        tempo_atuacao_min: int | None = None,
        tempo_atuacao_max: int | None = None,
        ja_recebeu_treinamento_opcoes: list[str] | None = None,
        ja_recebeu_treinamento_proporcao: list[int] | None = None,
    ):
        self.seed = seed
        self.n_personas = n_personas
        self.n_sessoes_por_persona = n_sessoes_por_persona
        self.llm_provider = llm_provider
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

        # Atributos demográficos (instância sobrescreve classe se fornecido)
        self.idade_min = idade_min if idade_min is not None else self.IDADE_MIN
        self.idade_max = idade_max if idade_max is not None else self.IDADE_MAX
        self.sexo_opcoes = sexo_opcoes or self.SEXO_OPCOES
        self.sexos_proporcao = sexos_proporcao or self.SEXOS_PROPORCAO
        self.setor_trabalha_opcoes = setor_trabalha_opcoes or self.SETOR_TRABALHA_OPCOES
        self.setor_trabalha_proporcao = (
            setor_trabalha_proporcao or self.SETOR_TRABALHA_PROPORCAO
        )
        self.porte_org_opcoes = porte_org_opcoes or self.PORTE_ORG_OPCOES
        self.porte_org_proporcao = porte_org_proporcao or self.PORTE_ORG_PROPORCAO
        self.funcao_cargos_opcoes = funcao_cargos_opcoes or self.FUNCAO_CARGOS_OPCOES
        self.funcao_cargos_proporcao = (
            funcao_cargos_proporcao or self.FUNCAO_CARGOS_PROPORCAO
        )
        self.tempo_atuacao_min = (
            tempo_atuacao_min
            if tempo_atuacao_min is not None
            else self.TEMPO_ATUACAO_MIN
        )
        self.tempo_atuacao_max = (
            tempo_atuacao_max
            if tempo_atuacao_max is not None
            else self.TEMPO_ATUACAO_MAX
        )
        self.ja_recebeu_treinamento_opcoes = (
            ja_recebeu_treinamento_opcoes or self.JA_RECEBEU_TREINAMENTO_OPCOES
        )
        self.ja_recebeu_treinamento_proporcao = (
            ja_recebeu_treinamento_proporcao or self.JA_RECEBEU_TREINAMENTO_PROPORCAO
        )

        # Fix SSL
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    # =========================================================================
    # 1. Configuração do LLM
    # =========================================================================

    def get_response(self, prompt, seed=42, temperature=0.7, num_predict=2048):
        if self.llm_provider == "openai":
            return self._get_response_openai(prompt, temperature=temperature)
        return self._get_response_ollama(
            prompt, seed=seed, temperature=temperature, num_predict=num_predict
        )

    def _get_response_ollama(self, prompt, seed=42, temperature=0.7, num_predict=2048):
        client = Client(host=self.ollama_host)
        response = client.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "seed": seed,
                "temperature": temperature,
                "num_predict": num_predict,
                # "num_ctx": 8192,
                # "num_batch": 512,
                # "top_k": 10,
                # "top_p": 0.9,
                # "repeat_penalty": 1.0,
            },
        )
        return response.message.content

    def _get_response_openai(self, prompt, temperature=0.7):
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, _enforce_credentials=False)
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    # =========================================================================
    # 2. Formatação dos dados reais
    # =========================================================================

    def formatar_dataset_real(self, path_csv):
        COLUNAS_ESPERADAS = [
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
            "idade",
            "sexo",
            "setor_trabalha",
            "porte_org",
            "funcao_cargo",
            "tempo_atuacao_org_anos",
            "ja_recebeu_treinamento",
        ]
        COLS_CATEGORICAS = {
            "sexo",
            "setor_trabalha",
            "porte_org",
            "funcao_cargo",
            "ja_recebeu_treinamento",
        }

        # Detectar formato: tenta leitura padrão primeiro
        df_real = pd.read_csv(path_csv, sep=",", engine="python", skipinitialspace=True)

        # Se o header não bate, tenta formato legado (sep='",'")
        if not set(COLUNAS_ESPERADAS).issubset(df_real.columns):
            novas_colunas = ["datahora_concorda"] + COLUNAS_ESPERADAS
            df_real = pd.read_csv(
                path_csv, sep='","', engine="python", names=novas_colunas, skiprows=1
            )
            obj_cols = df_real.select_dtypes(include="object").columns
            df_real[obj_cols] = df_real[obj_cols].apply(
                lambda col: col.str.strip().str.strip('"')
            )
            df_real = df_real.drop(columns=["datahora_concorda"], errors="ignore")

        # Converter colunas numéricas
        for col in df_real.columns:
            if col not in COLS_CATEGORICAS:
                df_real[col] = pd.to_numeric(df_real[col], errors="coerce")

        # Limpar strings categóricas
        for col in COLS_CATEGORICAS:
            if col in df_real.columns:
                df_real[col] = df_real[col].astype(str).str.strip().str.strip('"')

        # Remover linhas completamente vazias
        df_real = df_real.dropna(how="all")

        return df_real

    # =========================================================================
    # 3. Geração de personas
    # =========================================================================

    @staticmethod
    def _escolher(opcoes, pesos, usar_pesos):
        if usar_pesos:
            return random.choices(opcoes, weights=pesos, k=1)[0]
        return random.choice(opcoes)

    def gerar_personas_df(self, n=None, seed=None, usar_pesos=True):
        n = n or self.n_personas
        seed = seed or self.seed
        random.seed(seed)
        personas = []
        for i in range(1, n + 1):
            personas.append(
                {
                    "persona_id": f"HDT_{i:03d}",
                    "idade": random.randint(self.idade_min, self.idade_max),
                    "sexo": self._escolher(
                        self.sexo_opcoes, self.sexos_proporcao, usar_pesos
                    ),
                    "setor_trabalha": self._escolher(
                        self.setor_trabalha_opcoes,
                        self.setor_trabalha_proporcao,
                        usar_pesos,
                    ),
                    "porte_org": self._escolher(
                        self.porte_org_opcoes, self.porte_org_proporcao, usar_pesos
                    ),
                    "funcao_cargo": self._escolher(
                        self.funcao_cargos_opcoes,
                        self.funcao_cargos_proporcao,
                        usar_pesos,
                    ),
                    "tempo_atuacao_org_anos": random.randint(
                        self.tempo_atuacao_min, self.tempo_atuacao_max
                    ),
                    "ja_recebeu_treinamento": self._escolher(
                        self.ja_recebeu_treinamento_opcoes,
                        self.ja_recebeu_treinamento_proporcao,
                        usar_pesos,
                    ),
                    "pais": "Brasil",
                }
            )
        return pd.DataFrame(personas)

    def sobrescrever_com_distribuicao_real(self, personas_df, df_real):
        """Sobrescreve atributos das personas com amostragem da distribuição real."""
        rng = np.random.default_rng(self.seed)
        personas_df["idade"] = self._sample_from_real(
            df_real,
            "idade",
            len(personas_df),
            rng,
            numeric=True,
            min_value=18,
            max_value=70,
        )
        personas_df["sexo"] = self._sample_from_real(
            df_real, "sexo", len(personas_df), rng
        )
        personas_df["setor_trabalha"] = self._sample_from_real(
            df_real, "setor_trabalha", len(personas_df), rng
        )
        personas_df["porte_org"] = self._sample_from_real(
            df_real, "porte_org", len(personas_df), rng
        )
        personas_df["funcao_cargo"] = self._sample_from_real(
            df_real, "funcao_cargo", len(personas_df), rng
        )
        personas_df["tempo_atuacao_org_anos"] = self._sample_from_real(
            df_real, "tempo_atuacao_org_anos", len(personas_df), rng, numeric=True
        )
        personas_df["ja_recebeu_treinamento"] = self._sample_from_real(
            df_real, "ja_recebeu_treinamento", len(personas_df), rng
        )
        return personas_df

    @staticmethod
    def _sample_from_real(
        df_real,
        col,
        n,
        rng,
        numeric=False,
        min_value=None,
        max_value=None,
        round_numeric=True,
    ):
        s: pd.Series = df_real[col].dropna()  # type: ignore[assignment]
        if numeric:
            numeric_s: pd.Series = pd.to_numeric(s, errors="coerce").dropna()  # type: ignore[assignment]
            values = numeric_s.values
            if len(values) == 0:
                low = min_value if min_value is not None else 0
                high = max_value if max_value is not None else 100
                return rng.integers(low, high, size=n)
            sampled = rng.choice(values, size=n, replace=True)
            if round_numeric:
                sampled = np.rint(sampled).astype(int)
            if min_value is not None or max_value is not None:
                sampled = np.clip(
                    sampled,
                    min_value if min_value is not None else sampled.min(),
                    max_value if max_value is not None else sampled.max(),
                )
            return sampled
        if len(s) == 0:
            return np.array([None] * n)
        probs = s.value_counts(normalize=True)
        return rng.choice(probs.index, size=n, p=probs.values)

    # =========================================================================
    # 4. Prompt e geração de respostas
    # =========================================================================

    def calcular_target_probs(self, df_real):
        LIKERT = range(1, 8)
        return {
            q["item"]: (
                df_real[q["item"]]
                .value_counts(normalize=True)
                .reindex(LIKERT, fill_value=0)
                .round(3)
                .to_dict()
            )
            for q in self.QUESTIONARIO_7S
        }

    def montar_prompt_hdt(self, persona, target_probs):
        itens = "\n".join(
            f"- {q['item']} ({q['dimensao']}): {q['afirmacao']}"
            for q in self.QUESTIONARIO_7S
        )
        return f"""Você é um Human Digital Twin usado em uma pesquisa científica sobre proteção tecnológica organizacional no Brasil.

Responda como uma pessoa real com o perfil abaixo, considerando percepções, limitações, experiência profissional e contexto organizacional. Não responda como assistente de IA.

Perfil do respondente sintético:
{json.dumps(persona, ensure_ascii=False, indent=2)}

Questionário adaptado do modelo 7S de Proteção Tecnológica Organizacional:
{itens}

Use escala Likert de 1 a 7:
1 = discordo totalmente; 2 = discordo; 3 = discordo parcialmente; 4 = neutro; 5 = concordo parcialmente; 6 = concordo; 7 = concordo totalmente.

Regras:
- Responda com variabilidade compatível com o perfil.
- Considere que empresas maiores e perfis com maior envolvimento em segurança tendem a perceber controles mais maduros, mas não torne isso determinístico.
- Retorne somente JSON válido, sem markdown, sem comentários e sem texto fora do JSON.
- NÃO use quebras de linha dentro de valores de string.
- O JSON deve ter exatamente as chaves: persona_id, respostas, justificativa_curta.
- respostas deve mapear cada item para um número inteiro entre 1 e 7.

Distribuição-alvo aproximada por item:
{target_probs}""".strip()

    @staticmethod
    def _extrair_json(texto):
        texto = texto.strip()
        # Remove blocos markdown ```json ... ```
        texto = re.sub(r"```(?:json)?\s*", "", texto).strip().rstrip("`")
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            pass
        # Tenta extrair o maior bloco {...} do texto
        match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Nenhum JSON encontrado na resposta: {texto[:200]}")
        candidato = match.group(0)
        try:
            return json.loads(candidato)
        except json.JSONDecodeError:
            pass
        # Fallback: remove quebras de linha dentro de strings e tenta novamente
        candidato_limpo = re.sub(r'(?<=":)\s*"([^"]*)"', lambda m: '"' + m.group(1).replace('\n', ' ').replace('\r', '') + '"', candidato)
        return json.loads(candidato_limpo)

    def _normalizar_respostas(self, resultado, persona):
        respostas = resultado.get("respostas", {})
        linha = {**persona}
        linha["justificativa_curta"] = resultado.get("justificativa_curta", "")
        for q in self.QUESTIONARIO_7S:
            valor = respostas.get(q["item"])
            linha[q["item"]] = int(valor) if str(valor).isdigit() else None
        return linha

    def responder_persona(self, persona, target_probs, sessao=1):
        prompt = self.montar_prompt_hdt(persona, target_probs)
        seed = self.seed + sessao * 1000 + int(persona["persona_id"].split("_")[1])
        texto = self.get_response(prompt, seed=seed, temperature=0.8)
        resultado = self._extrair_json(texto)
        return self._normalizar_respostas(resultado, persona)

    # =========================================================================
    # 5. Execução completa
    # =========================================================================

    def executar(
        self, path_csv_real="inputs/questionario_real.csv", output_dir="outputs"
    ):
        """Pipeline completo: formata dados reais, gera personas, coleta respostas e salva."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Formatar dados reais
        df_real = self.formatar_dataset_real(path_csv_real)
        df_real.to_csv("inputs/questionario_real_formatado.csv", index=False)

        # Gerar personas
        personas_df = self.gerar_personas_df()
        personas_df = self.sobrescrever_com_distribuicao_real(personas_df, df_real)
        personas = personas_df.to_dict(orient="records")

        # Calcular distribuição-alvo
        target_probs = self.calcular_target_probs(df_real)

        # Gerar respostas
        resultados = []
        for sessao in range(1, self.n_sessoes_por_persona + 1):
            for persona in tqdm(personas, desc="Gerando respostas"):
                try:
                    linha = self.responder_persona(persona, target_probs, sessao=sessao)
                    linha["sessao"] = sessao
                    resultados.append(linha)
                except Exception as e:
                    print(f"Erro em {persona['persona_id']} sessão {sessao}: {e}")

        df_respostas = pd.DataFrame(resultados)

        # Salvar
        personas_df.to_csv(output_dir / "hdt_personas.csv", index=False)
        pd.DataFrame(self.QUESTIONARIO_7S).to_csv(
            output_dir / "questionario_7s.csv", index=False
        )
        df_respostas.to_csv(output_dir / "hdt_respostas_7s.csv", index=False)

        # Gerar CSV no formato do questionário sintético
        itens = [q["item"] for q in self.QUESTIONARIO_7S]
        cols_export = ["persona_id", "sexo", "idade"] + itens
        cols_export = [c for c in cols_export if c in df_respostas.columns]
        df_export: pd.DataFrame = df_respostas[cols_export].rename(
            columns={"persona_id": "Perfil"}
        )  # type: ignore[assignment]
        df_export.to_csv(
            f"outputs/questionario_sintetico_{len(df_respostas)}.csv", index=False
        )

        print(f"✅ Geração concluída: {len(df_respostas)} respostas salvas.")
        return df_real, personas_df, df_respostas
