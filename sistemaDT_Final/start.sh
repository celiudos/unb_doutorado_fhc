#!/bin/zsh
# Sobe o Digital Twin em http://127.0.0.1:8000 (de qualquer diretório).
cd "$(dirname "$0")" || exit 1
lsof -ti :8000 | xargs kill 2>/dev/null
sleep 1
nohup .venv/bin/uvicorn twin_api.main:app --host 127.0.0.1 --port 8000 > var/uvicorn.log 2>&1 &
sleep 2
if curl -s http://127.0.0.1:8000/health > /dev/null; then
  echo "Digital Twin no ar: http://127.0.0.1:8000 (PID $(lsof -ti :8000))"
  echo "Para parar: ./stop.sh"
else
  echo "Falhou ao subir — veja var/uvicorn.log"
  exit 1
fi
