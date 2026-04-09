#!/usr/bin/env bash
# Start the API from the correct directory (fixes "Could not import module main").
# Frees PORT if something is already listening (fixes "Address already in use").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"

pids="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
if [[ -n "${pids}" ]]; then
  echo "Port ${PORT} is in use; stopping PID(s): ${pids}"
  kill ${pids} 2>/dev/null || true
  sleep 1
fi

if [[ -f "${ROOT}/../venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/../venv/bin/activate"
elif [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

echo "Starting uvicorn from: ${ROOT}"
exec uvicorn main:app --host 127.0.0.1 --port "${PORT}" --reload
