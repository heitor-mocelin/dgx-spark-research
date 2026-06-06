#!/usr/bin/env bash
# Render prometheus/prometheus.yml from the .tmpl using values in .env.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "ERROR: .env not found. Run: cp .env.example .env && edit it."; exit 1; }
# shellcheck disable=SC1091
set -a; . ./.env; set +a

: "${DGX_HOST:?set DGX_HOST in .env}"
: "${DGX_NAME:?set DGX_NAME in .env}"

if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst not found (install 'gettext' / 'gettext-base')."; exit 1
fi

envsubst '${DGX_HOST} ${DGX_NAME}' < prometheus/prometheus.yml.tmpl > prometheus/prometheus.yml
echo "Rendered prometheus/prometheus.yml  (DGX_HOST=$DGX_HOST, DGX_NAME=$DGX_NAME)"
