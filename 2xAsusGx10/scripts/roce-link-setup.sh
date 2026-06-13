#!/usr/bin/env bash
set -euo pipefail
HN=$(hostname)
case "$HN" in
  *d1c6*) OCT=10 ; ROLE="head" ;;
  *52cc*) OCT=11 ; ROLE="worker" ;;
  *) echo "Unknown host $HN — set OCT manually (10=head .210, 11=worker .211)"; exit 1 ;;
esac
PRI=""; SEC=""
for n in $(ls /sys/class/net | grep -E '^en.*s0f[01]np[01]$'); do
  [ "$(cat /sys/class/net/$n/carrier 2>/dev/null)" = "1" ] || continue
  case "$n" in
    enp1*)   PRI="$n" ;;
    enP2p1*) SEC="$n" ;;
  esac
done
echo "host=$HN role=$ROLE  railA(primary x4)=$PRI  railB(secondary x4)=$SEC"
[ -n "$PRI" ] && [ -n "$SEC" ] || { echo "ERROR: could not find both live netdevs (is the cable linked?)"; exit 1; }
add_rail(){ # name ifname cidr
  nmcli con add type ethernet con-name "$1" ifname "$2" ipv4.method manual \
        ipv4.addresses "$3" ipv6.method disabled connection.autoconnect yes 2>/dev/null \
    || nmcli con mod "$1" connection.interface-name "$2" ipv4.method manual ipv4.addresses "$3" ipv6.method disabled
  nmcli con up "$1"
}
add_rail roce-railA "$PRI" "192.168.100.$OCT/24"
add_rail roce-railB "$SEC" "192.168.101.$OCT/24"
echo "=== result ==="
ip -br addr show "$PRI"; ip -br addr show "$SEC"
echo "Done. railA=192.168.100.$OCT  railB=192.168.101.$OCT"
