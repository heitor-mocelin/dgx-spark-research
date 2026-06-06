#!/usr/bin/env bash
# OPTIONAL: create a Debian 12 LXC on a Proxmox VE node to host the monitoring stack.
# Run this ON a Proxmox node (as root). It creates an unprivileged CT with Docker,
# then you clone this repo inside it and run scripts/setup.sh.
#
# Usage:  ./proxmox-create-lxc.sh <vmid> <ip-cidr> <gateway> [storage] [bridge]
# Example: ./proxmox-create-lxc.sh 210 192.168.1.60/24 192.168.1.1 local-lvm vmbr0
set -euo pipefail

VMID="${1:?usage: proxmox-create-lxc.sh <vmid> <ip-cidr> <gateway> [storage] [bridge]}"
IPCIDR="${2:?missing ip-cidr, e.g. 192.168.1.60/24}"
GW="${3:?missing gateway, e.g. 192.168.1.1}"
STORAGE="${4:-local-lvm}"
BRIDGE="${5:-vmbr0}"
TEMPLATE_STORE="${TEMPLATE_STORE:-local}"
TEMPLATE="debian-12-standard_12.7-1_amd64.tar.zst"   # adjust arch if your node is arm64

echo "==> Ensuring template $TEMPLATE is available"
pveam update >/dev/null 2>&1 || true
pveam list "$TEMPLATE_STORE" | grep -q "$TEMPLATE" || pveam download "$TEMPLATE_STORE" "$TEMPLATE"

echo "==> Creating CT $VMID"
pct create "$VMID" "$TEMPLATE_STORE:vztmpl/$TEMPLATE" \
  --hostname dgx-monitoring \
  --cores 2 --memory 2048 --swap 512 \
  --rootfs "$STORAGE:8" \
  --net0 "name=eth0,bridge=$BRIDGE,ip=$IPCIDR,gw=$GW" \
  --features nesting=1 \
  --unprivileged 1 \
  --onboot 1 --start 1

echo "==> Waiting for network"; sleep 5

echo "==> Installing Docker inside CT $VMID"
pct exec "$VMID" -- bash -lc '
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl git gettext-base >/dev/null
  install -m0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin >/dev/null
  systemctl enable --now docker
'

cat <<EOF

CT $VMID is up with Docker. Finish inside it:

  pct enter $VMID
  git clone <your-fork-url> dgx-spark-monitoring && cd dgx-spark-monitoring
  cp .env.example .env && \$EDITOR .env      # set DGX_HOST / DGX_NAME
  ./scripts/setup.sh

Grafana will be at http://$(echo "$IPCIDR" | cut -d/ -f1):3000
EOF
