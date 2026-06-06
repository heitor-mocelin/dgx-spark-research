#!/usr/bin/env bash
# One-shot quickstart for the MONITORING host (Prometheus + Grafana).
# Renders the Prometheus config from .env, then brings the stack up with Docker Compose.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/3  Checking prerequisites"
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not installed."; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: 'docker compose' plugin not available."; exit 1; }
[ -f .env ] || { echo "ERROR: .env missing. Run: cp .env.example .env  (then edit DGX_HOST/DGX_NAME)."; exit 1; }

echo "==> 2/3  Rendering Prometheus config"
bash scripts/render-prometheus.sh

echo "==> 3/3  Starting Prometheus + Grafana"
docker compose up -d

# shellcheck disable=SC1091
set -a; . ./.env; set +a
echo
echo "Done. Open Grafana:  http://localhost:3000   (user: ${GRAFANA_ADMIN_USER:-admin})"
echo "Dashboard:           Dashboards -> 'DGX Spark — GB10 Observability'"
echo "Prometheus targets:  http://localhost:9090/targets   (verify dcgm/node/vllm are UP)"
echo
echo "Next: on the DGX, run  'cd dgx && docker compose up -d'  to start the exporters."
