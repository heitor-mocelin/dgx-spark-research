#!/usr/bin/env bash
# Run ON the DGX Spark. Verifies Docker + NVIDIA runtime, then starts the exporters.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Checking Docker"
command -v docker >/dev/null 2>&1 || { echo "ERROR: install Docker first (https://docs.docker.com/engine/install/)."; exit 1; }

echo "==> Checking NVIDIA Container Toolkit (needed for dcgm-exporter)"
if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
  cat <<'EOF'
WARNING: the 'nvidia' Docker runtime was not detected.
  Install the NVIDIA Container Toolkit, then re-run:
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  (dcgm-exporter will fail to start without it; node-exporter and cAdvisor will still work.)
EOF
fi

echo "==> Quick GPU sanity (nvidia-smi)"
nvidia-smi -L || echo "  (nvidia-smi not found on host — that's fine if drivers live elsewhere)"

echo "==> Starting exporters (node-exporter, dcgm-exporter, cAdvisor)"
docker compose up -d
echo
echo "Verify locally:"
echo "  curl -s localhost:9100/metrics | head      # node_exporter"
echo "  curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL   # dcgm-exporter"
echo "  curl -s localhost:8080/healthz             # cAdvisor"
