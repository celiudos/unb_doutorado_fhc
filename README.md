# Gerador de Personas Sintéticas - HDT 7S

Interface para geração de dados sintéticos de proteção tecnológica organizacional usando Human Digital Twins (HDTs) e o modelo 7S.

## Como rodar

### 1. Instalar dependências

```bash
pip install openai ollama gradio numpy pandas matplotlib certifi tqdm python-dotenv
```

### 2. Configurar variável de ambiente (opcional)

Copie o `.env.example` para `.env` e preencha sua chave da OpenAI:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

A chave será carregada automaticamente na interface. Caso não configure o `.env`, você pode inserir a key diretamente na UI.

### 3. Executar

Abra e execute o notebook:

```
00_ui_gerar_dados_sinteticos.ipynb
```

A interface Gradio será aberta em `http://127.0.0.1:7860`.
