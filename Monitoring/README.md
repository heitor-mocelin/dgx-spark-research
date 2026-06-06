# DGX Spark — GB10 Observability

A complete, self-hosted monitoring stack and Grafana dashboard for the **NVIDIA DGX Spark / GB10 Grace-Blackwell** (and similar Grace-Blackwell systems). One screen for **GPU, Grace CPU, unified memory, thermals & power**, plus first-class **LLM inference performance** for **vLLM**, **llama.cpp** and **ollama** — including which model is loaded, KV-cache pressure, throughput and latency.

Built to be **portable** (template variables, no hardcoded IPs) and **honest** (panels say *"🔴 Offline / not running"* instead of a blank *"No data"* when a service is down).

> Works on any host that can scrape a DGX Spark. The GB10 is arm64 and uses **unified memory** (CPU + GPU share one LPDDR5 pool) — the dashboard treats memory accordingly. See [GB10 notes](#gb10-unified-memory-notes).

<!-- Add a screenshot here once deployed: ![dashboard](images/overview.png) -->

---

## Table of contents
- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [Step 2 — DGX exporters](#step-2--dgx-exporters)
- [Optional — run the stack in a Proxmox LXC](#optional--run-the-stack-in-a-proxmox-lxc)
- [Wiring your inference servers](#wiring-your-inference-servers)
- [Import into an existing Grafana](#import-into-an-existing-grafana)
- [Dashboard tour](#dashboard-tour)
- [How the offline-aware tiles work](#how-the-offline-aware-tiles-work)
- [Customizing](#customizing)
- [Troubleshooting](#troubleshooting)
- [GB10 unified-memory notes](#gb10-unified-memory-notes)
- [Repo layout](#repo-layout)
- [License](#license)

---

## What you get

- **One dashboard, six areas:** Overview & Health · GPU (DCGM) · Grace CPU & Unified Memory · vLLM · llama.cpp · ollama · Storage & Network.
- **GPU telemetry** from `dcgm-exporter`: SM/mem-copy utilisation, power & energy, SM/mem clocks, GPU & memory temperature, framebuffer (VRAM where reported), XID errors and throttle violations.
- **LLM performance** that actually matters: aggregate tok/s (generation + prefill), concurrent/queued requests, **KV-cache utilisation**, TTFT & end-to-end latency percentiles, prefix-cache hit rate, and **the loaded model + its memory**.
- **Offline-aware tiles:** a per-engine 🟢/🔴 status, and KPI tiles that print a friendly message when their exporter is genuinely down — without false alarms on transient scrape gaps.
- **Reproducible & scriptable:** `docker compose up` for the stack, a one-liner for the DGX exporters, everything templated from a single `.env`.

## Architecture

```
            ┌──────────────────────── DGX Spark (GB10, arm64) ────────────────────────┐
            │                                                                          │
            │  node-exporter   dcgm-exporter   cAdvisor      vLLM      llama-server    │
            │     :9100           :9400          :8080       :8000        :8090        │
            │       │               │             │            │            │          │
            └───────┼───────────────┼─────────────┼────────────┼────────────┼─────────┘
                    │   scrape (pull over HTTP, every 5–10s)    │            │
                    ▼               ▼             ▼            ▼            ▼
            ┌───────────────────────────── Monitoring host ─────────────────────────────┐
            │   Prometheus (:9090)  ──────────────►  Grafana (:3000)                     │
            │   stores time-series                   dashboard + provisioned datasource  │
            └───────────────────────────────────────────────────────────────────────────┘
```

- **Exporters** run *on the DGX* and expose `/metrics`.
- **Prometheus** (on any host) pulls them and stores the series.
- **Grafana** queries Prometheus and renders the dashboard.

The monitoring host can be a Proxmox LXC, a spare mini-PC, or even the DGX itself.

## Prerequisites

- A **DGX Spark / GB10** (or similar NVIDIA system) with Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed (needed for `dcgm-exporter`).
- A **monitoring host** with Docker + the Compose plugin, able to reach the DGX over the network.
- That's it — no Kubernetes, no cloud.

## Quick start (Docker Compose)

On the **monitoring host**:

```bash
git clone https://github.com/<you>/dgx-spark-monitoring.git
cd dgx-spark-monitoring

cp .env.example .env
$EDITOR .env          # set DGX_HOST (the DGX's IP) and DGX_NAME (a label, e.g. "dgx-spark")

./scripts/setup.sh    # renders prometheus.yml from .env, then `docker compose up -d`
```

Open **http://localhost:3000** (login from `.env`, default `admin`/`changeme`) → **Dashboards → "DGX Spark — GB10 Observability"**.

Check **http://localhost:9090/targets** — once the DGX exporters are up (next step), `node`, `dcgm` and `vllm` turn green.

## Step 2 — DGX exporters

On the **DGX Spark** itself:

```bash
cd dgx-spark-monitoring/dgx
./install-exporters.sh        # verifies Docker + nvidia runtime, then `docker compose up -d`
```

This starts three containers:

| Exporter        | Port  | Provides                                            |
|-----------------|-------|-----------------------------------------------------|
| `node-exporter` | 9100  | CPU, memory, disk, network, thermals                |
| `dcgm-exporter` | 9400  | GPU util, power, temp, clocks, framebuffer, errors  |
| `cadvisor`      | 8080  | per-container CPU/RAM (attribute usage to engines)  |

Verify on the DGX:

```bash
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL    # GPU is exporting
```

> **Firewall:** make sure ports `9100/9400/8080` (and your inference ports) are reachable from the monitoring host.

## Optional — run the stack in a Proxmox LXC

If your monitoring host is a Proxmox VE cluster, `scripts/proxmox-create-lxc.sh` creates a Debian 12 container with Docker pre-installed. Run it **on a Proxmox node, as root**:

```bash
./scripts/proxmox-create-lxc.sh 210 192.168.1.60/24 192.168.1.1 local-lvm vmbr0
#                                │   │               │           │         └ bridge
#                                │   │               │           └ storage
#                                │   │               └ gateway
#                                │   └ CT IP/CIDR
#                                └ VMID
```

Then `pct enter 210`, clone the repo inside, and run `./scripts/setup.sh` as in the quick start.

## Wiring your inference servers

The dashboard expects these Prometheus jobs (already in `prometheus/prometheus.yml.tmpl`). Start each engine with metrics enabled:

| Engine     | Job        | Start with                                                                 |
|------------|------------|----------------------------------------------------------------------------|
| **vLLM**   | `vllm`     | metrics are **on by default** at `:8000/metrics` (OpenAI server)           |
| **llama.cpp** | `llamacpp` | `llama-server --metrics --port 8090 -m <model.gguf> …`                  |
| **ollama** | `ollama`   | optional — run an ollama Prometheus exporter and point job `ollama` at it  |

If you don't run one of them, that section simply shows **🔴 Offline** / *not running* — no errors, nothing to configure away.

> Already running an engine on a non-default port? Edit `prometheus/prometheus.yml.tmpl`, re-run `scripts/render-prometheus.sh`, then `curl -X POST http://localhost:9090/-/reload`.

## Import into an existing Grafana

Already have Prometheus + Grafana? Skip the stack and just import the dashboard:

1. **Grafana → Dashboards → New → Import.**
2. Upload `dashboards/dgx-spark.json` (or paste it).
3. Select your Prometheus datasource when prompted (the `$datasource` variable).
4. Pick your host in the **DGX host** dropdown (`$instance`).

For the `$instance` values to appear, your Prometheus must expose a consistent `instance` label across the `node`/`dcgm`/`vllm`/… jobs — see the relabel/labels approach in `prometheus/prometheus.yml.tmpl`.

## Dashboard tour

- **📊 Overview & Health** — per-exporter 🟢/🔴 status, plus headline GPU util/temp/power, unified-memory %, CPU % and vLLM tok/s.
- **🎮 GPU — NVIDIA Blackwell (DCGM)** — utilisation & mem-copy, power & energy rate, clocks, GPU/memory temps, framebuffer (VRAM), and an errors/throttling panel (XID, PCIe replay, power/thermal violations).
- **🧠 Grace CPU & Unified Memory** — CPU by mode, load average, the 128 GB unified pool (used/cache/free) and a memory-pressure gauge.
- **🚀 vLLM** — server status, served model, KV-cache gauge, prefix-cache hit, process RAM; throughput, concurrency, TTFT and E2E latency percentiles, process CPU/RAM.
- **🦙 llama.cpp** — server status, KV-cache, requests processing/deferred; throughput (generation + prefill).
- **🤖 ollama** *(collapsed)* — status, models loaded, per-model RAM.
- **💾 Storage & 🌐 Network** *(collapsed)* — NVMe I/O, filesystem usage, network throughput and errors.

## How the offline-aware tiles work

Two mechanisms keep the board readable when something is off:

1. **Status tiles** read the Prometheus `up{job="…"}` series (always present: `0`=down, `1`=up) and map it to **🟢 Running / 🔴 Offline**.
2. **KPI tiles** use a *down-scoped sentinel*:

   ```promql
   (<your-metric>) or (vector(-1) and on() (up{job="vllm"} == 0))
   ```

   The tile shows the real value when it exists; emits `-1` **only when the exporter is genuinely down** (`up==0`), which a value-mapping renders as *"vLLM off"*. A momentary scrape gap while `up==1` falls through to the panel's *"No data"* text instead of a false "offline". Idle-but-running engines correctly read `0`.

This is why you can leave llama.cpp stopped and the board stays clean and truthful.

## Customizing

- **Different host / multiple DGX boxes:** the **DGX host** (`$instance`) dropdown lists every host with a `node` job. Add more targets in `prometheus.yml.tmpl` with distinct `instance` labels.
- **Multiple GPUs:** GPU panels already split by the DCGM `gpu` label.
- **Edit and re-generate:** the dashboard is generated by `generate_dashboard.py` (pure Python, no deps) — tweak panels there and run `python3 generate_dashboard.py` to rewrite `dashboards/dgx-spark.json`.
- **Scrape interval:** change `scrape_interval` in `prometheus.yml.tmpl` (5s gives smoother GPU graphs at higher TSDB cost).

## Troubleshooting

| Symptom | Fix |
|---|---|
| **`$instance` dropdown is empty** | `node-exporter` isn't being scraped, or has no consistent `instance` label. Check `http://<mon>:9090/targets` and that `prometheus.yml` was rendered from your `.env`. |
| **All GPU panels say "No data"** | `dcgm-exporter` down or unreachable. On the DGX: `docker logs dcgm-exporter`; confirm the **nvidia runtime** is installed; `curl localhost:9400/metrics`. |
| **vLLM panels empty but vLLM is running** | vLLM metrics live at `:8000/metrics`. Confirm the `vllm` target is UP and the port matches your server. |
| **llama.cpp empty** | Start `llama-server` with `--metrics` on port `8090` (or edit the job). It exposes no model-name label by design. |
| **Framebuffer/VRAM panel empty** | Expected on unified-memory GB10 — see below. |
| **Prometheus target DOWN: connection refused** | Firewall, or the exporter/engine isn't running on that port. |

## GB10 unified-memory notes

The GB10 Grace-Blackwell shares a single **128 GB LPDDR5 pool** between the Grace CPU and the Blackwell GPU — there is **no dedicated VRAM**. So:

- Track "GPU memory" via the **Grace CPU & Unified Memory** row (it's the same physical pool).
- `dcgm-exporter` framebuffer metrics (`DCGM_FI_DEV_FB_USED/FREE`) may report the GPU's carved slice or nothing, depending on driver/exporter version — the VRAM panel is intentionally tolerant and shows a *No data* message when absent.
- A separate VRAM panel only becomes meaningful on systems with a **discrete** GPU.

## Repo layout

```
dgx-spark-monitoring/
├── docker-compose.yml              # Prometheus + Grafana (monitoring host)
├── .env.example                    # DGX_HOST / DGX_NAME / Grafana creds
├── prometheus/
│   └── prometheus.yml.tmpl         # scrape jobs (rendered to prometheus.yml)
├── grafana/provisioning/
│   ├── datasources/datasource.yml  # auto-add Prometheus
│   └── dashboards/dashboards.yml   # auto-load the dashboard
├── dashboards/
│   └── dgx-spark.json              # the dashboard (generated)
├── dgx/
│   ├── docker-compose.yml          # node-exporter + dcgm-exporter + cAdvisor (run on DGX)
│   └── install-exporters.sh
├── scripts/
│   ├── setup.sh                    # render config + start the stack
│   ├── render-prometheus.sh        # envsubst .tmpl -> prometheus.yml
│   └── proxmox-create-lxc.sh       # optional: Debian+Docker LXC on Proxmox
├── generate_dashboard.py           # regenerates dashboards/dgx-spark.json
└── LICENSE                         # MIT
```

## License

MIT — see [LICENSE](LICENSE). Contributions welcome: open an issue or PR with your DGX/Grace-Blackwell tweaks, extra exporters, or dashboard improvements.
