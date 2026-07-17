#!/bin/zsh
# Para o Digital Twin (porta 8000).
if lsof -ti :8000 > /dev/null; then
  lsof -ti :8000 | xargs kill
  echo "Servidor parado."
else
  echo "Nada rodando na porta 8000."
fi
