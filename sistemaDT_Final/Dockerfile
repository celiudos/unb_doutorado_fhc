FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY twin_core ./twin_core
COPY twin_api ./twin_api
COPY twin_web ./twin_web
COPY dados ./dados
COPY scripts ./scripts

RUN pip install --no-cache-dir . && useradd -m twin && mkdir -p /app/var && chown -R twin /app/var
USER twin

ENV TWIN_VAR_DIR=/app/var
EXPOSE 8000
CMD ["sh", "-c", "python scripts/seed.py && uvicorn twin_api.main:app --host 0.0.0.0 --port 8000"]
