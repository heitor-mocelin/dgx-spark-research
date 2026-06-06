#!/usr/bin/env python3
"""Generate the community 'DGX Spark — GB10 Observability' Grafana dashboard.

Generic & portable: driven by two template variables (`$datasource`, `$instance`) so it works
on any DGX Spark / GB10 (or similar NVIDIA Grace-Blackwell) box without editing the JSON.
Reuses the offline-aware design (up-status tiles, down-scoped sentinels, friendly No-data text).

Expected Prometheus jobs (see prometheus/prometheus.yml in this repo):
  node      node_exporter            <host>:9100
  dcgm      dcgm-exporter            <host>:9400
  cadvisor  cAdvisor (optional)      <host>:8080
  vllm      vLLM /metrics            <host>:8000
  llamacpp  llama-server --metrics   <host>:8090
  ollama    ollama exporter          <host>:9778 (optional)
A relabel rule sets `instance` = bare host on every job, so one `$instance` drives all panels.
"""
import json

# Datasource is a template variable so the dashboard is portable.
DS = {"type": "prometheus", "uid": "${datasource}"}
_id = [0]
def nid():
    _id[0] += 1
    return _id[0]

# per-job selectors (instance unified by relabel in prometheus.yml)
I = 'instance="$instance"'
N = f'job="node",{I}'        # node_exporter
G = f'job="dcgm",{I}'        # dcgm-exporter
VL = f'job="vllm",{I}'
LC = f'job="llamacpp",{I}'
OL = f'job="ollama",{I}'
NETFLT = 'device!~"lo|veth.*|docker.*|br-.*|cni.*|flannel.*|cali.*|tap.*"'

T_PCT = [{"color": "green", "value": None}, {"color": "yellow", "value": 70}, {"color": "red", "value": 90}]
T_TEMP = [{"color": "green", "value": None}, {"color": "yellow", "value": 65}, {"color": "red", "value": 85}]
T_KV = [{"color": "green", "value": None}, {"color": "yellow", "value": 80}, {"color": "red", "value": 95}]
T_PWR = [{"color": "blue", "value": None}, {"color": "green", "value": 10}, {"color": "yellow", "value": 60}, {"color": "red", "value": 110}]

MAP_UP = [{"type": "value", "options": {
    "0": {"text": "🔴 Offline", "color": "red", "index": 0},
    "1": {"text": "🟢 Running", "color": "green", "index": 1}}}]
NOVAL = "⚪  No data — exporter / engine not running"

def OFF(text):
    return [{"type": "value", "options": {"-1": {"text": text, "color": "text", "index": 0}}}]

def SENT(expr):
    """Plain sentinel: empty -> -1. Use where absence == not running (e.g. cAdvisor container)."""
    return f'({expr}) or vector(-1)'

def SENTJ(expr, jobsel):
    """Down-scoped sentinel: -1 ONLY when up{jobsel}==0 (real outage), not on transient gaps."""
    return f'({expr}) or (vector(-1) and on() (up{{{jobsel}}} == 0))'


def tgt(expr, legend="", instant=False, refid="A", fmt=None):
    t = {"datasource": DS, "editorMode": "code", "expr": expr,
         "legendFormat": legend or "__auto", "range": not instant, "instant": instant, "refId": refid}
    if fmt:
        t["format"] = fmt
    return t


def ts(title, targets, x, y, w, h, unit="percent", mn=None, mx=None, desc="",
       stack=False, fill=10, draw="line", noValue=NOVAL):
    fc = {"drawStyle": draw, "lineInterpolation": "smooth", "lineWidth": 2, "fillOpacity": fill,
          "gradientMode": "opacity", "spanNulls": False, "showPoints": "never", "pointSize": 5,
          "stacking": {"mode": "normal" if stack else "none", "group": "A"},
          "axisCenteredZero": False, "axisColorMode": "text", "axisLabel": "", "axisPlacement": "auto",
          "barAlignment": 0, "thresholdsStyle": {"mode": "off"}}
    fd = {"color": {"mode": "palette-classic"}, "custom": fc, "unit": unit,
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}
    if mn is not None: fd["min"] = mn
    if mx is not None: fd["max"] = mx
    if noValue is not None: fd["noValue"] = noValue
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "fieldConfig": {"defaults": fd, "overrides": []},
            "options": {"legend": {"displayMode": "table", "placement": "bottom",
                        "calcs": ["lastNotNull", "max", "mean"], "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def stat(title, expr, x, y, w, h, unit="none", thresholds=None, desc="", decimals=None,
         text_mode="auto", color_mode="value", graph="area", mappings=None, noValue=None):
    steps = thresholds or [{"color": "green", "value": None}]
    defaults = {"color": {"mode": "thresholds"}, "unit": unit,
                "thresholds": {"mode": "absolute", "steps": steps}, "mappings": mappings or []}
    if noValue is not None: defaults["noValue"] = noValue
    d = {"id": nid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [tgt(expr)],
         "fieldConfig": {"defaults": defaults, "overrides": []},
         "options": {"colorMode": color_mode, "graphMode": graph, "justifyMode": "auto",
             "orientation": "auto", "textMode": text_mode,
             "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
             "wideLayout": True, "showPercentChange": False}}
    if decimals is not None: d["fieldConfig"]["defaults"]["decimals"] = decimals
    return d


def gauge(title, expr, x, y, w, h, unit="percent", thresholds=None, mx=100, desc="", noValue=NOVAL):
    steps = thresholds or T_KV
    defaults = {"color": {"mode": "thresholds"}, "unit": unit, "min": 0, "max": mx,
                "thresholds": {"mode": "absolute", "steps": steps}}
    if noValue is not None: defaults["noValue"] = noValue
    return {"id": nid(), "type": "gauge", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [tgt(expr)],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"orientation": "auto", "showThresholdLabels": False, "showThresholdMarkers": True,
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def text(title, content, x, y, w, h):
    return {"id": nid(), "type": "text", "title": title, "datasource": None,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": content}}


def model_table(title, targets, x, y, w, h, desc=""):
    return {"id": nid(), "type": "table", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "transformations": [{"id": "organize", "options": {
                "excludeByName": {"Time": True, "__name__": True, "job": True, "instance": True,
                                  "host": True, "engine": True, "Hostname": True},
                "renameByName": {"model": "Model", "model_name": "Model", "Value": "Value", "name": "Container"}}}],
            "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}}, "overrides": []},
            "options": {"showHeader": True, "cellHeight": "sm"}}


# ---- placement engine -----------------------------------------------------
panels = []
gy = [0]

def row_header(title, y, collapsed, children):
    return {"id": nid(), "type": "row", "title": title, "collapsed": collapsed,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": children}

def place(title, plist, collapsed=False):
    if collapsed:
        panels.append(row_header(title, gy[0], True, plist)); gy[0] += 1
    else:
        panels.append(row_header(title, gy[0], False, []))
        base = gy[0] + 1
        sect_h = max((p["gridPos"]["y"] + p["gridPos"]["h"]) for p in plist) if plist else 0
        for p in plist:
            p["gridPos"]["y"] += base
        panels.extend(plist); gy[0] = base + sect_h


# =========================================================================
# OVERVIEW & HEALTH
# =========================================================================
ov = []
# status tiles (up-based)
ov.append(stat("node_exporter", f'up{{{N}}}', 0, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("DCGM GPU", f'up{{{G}}}', 4, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("vLLM", f'up{{{VL}}}', 8, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("llama.cpp", f'up{{{LC}}}', 12, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("ollama", f'up{{{OL}}}', 16, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("cAdvisor", f'up{{job="cadvisor",{I}}}', 20, 0, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
# KPI tiles (offline-aware)
ov.append(stat("GPU Util", SENTJ(f'avg(DCGM_FI_DEV_GPU_UTIL{{{G}}})', G), 0, 4, 4, 4,
               unit="percent", thresholds=T_PCT, mappings=OFF("GPU off")))
ov.append(stat("GPU Temp", SENTJ(f'max(DCGM_FI_DEV_GPU_TEMP{{{G}}})', G), 4, 4, 4, 4,
               unit="celsius", thresholds=T_TEMP, mappings=OFF("GPU off")))
ov.append(stat("GPU Power", SENTJ(f'sum(DCGM_FI_DEV_POWER_USAGE{{{G}}})', G), 8, 4, 4, 4,
               unit="watt", thresholds=T_PWR, mappings=OFF("GPU off")))
ov.append(stat("Unified Mem", SENTJ(f'100*(1-node_memory_MemAvailable_bytes{{{N}}}/node_memory_MemTotal_bytes{{{N}}})', N),
               12, 4, 4, 4, unit="percent", thresholds=T_PCT, decimals=1, mappings=OFF("node off")))
ov.append(stat("CPU", SENTJ(f'100-(avg(rate(node_cpu_seconds_total{{{N},mode="idle"}}[5m]))*100)', N),
               16, 4, 4, 4, unit="percent", thresholds=T_PCT, decimals=1, mappings=OFF("node off")))
ov.append(stat("vLLM tok/s", SENTJ(f'sum(rate(vllm:generation_tokens_total{{{VL}}}[1m]))', VL), 20, 4, 4, 4,
               unit="none", decimals=0, color_mode="none", mappings=OFF("vLLM off")))
place("📊  Overview & Health", ov, collapsed=False)

# =========================================================================
# GPU — NVIDIA Blackwell (DCGM)
# =========================================================================
gp = []; yy = 0
gp.append(text("NVIDIA GB10 — DCGM", (
    "### 🎮 GPU — NVIDIA (DCGM exporter)\n"
    "Metrics from `dcgm-exporter` (job `dcgm`). On **GB10 the GPU shares the unified LPDDR5 memory** with the CPU, "
    "so framebuffer (VRAM) panels may report the carved GPU slice or be empty — see **Unified Memory**. "
    "Optional metrics (XID, throttle violations) appear when your DCGM field set includes them."
), 0, yy, 24, 3)); yy += 3
gp.append(ts("GPU Utilisation & Mem-Copy", [
    tgt(f'DCGM_FI_DEV_GPU_UTIL{{{G}}}', legend="SM util gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEM_COPY_UTIL{{{G}}}', legend="mem-copy gpu{{gpu}}", refid="B"),
], 0, yy, 12, 7, unit="percent", mn=0, mx=100))
gp.append(ts("GPU Power & Energy Rate", [
    tgt(f'DCGM_FI_DEV_POWER_USAGE{{{G}}}', legend="power W gpu{{gpu}}"),
    tgt(f'rate(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{{G}}}[1m])/1000', legend="energy rate W gpu{{gpu}}", refid="B"),
], 12, yy, 12, 7, unit="watt", fill=8)); yy += 7
gp.append(ts("GPU & Memory Temperature", [
    tgt(f'DCGM_FI_DEV_GPU_TEMP{{{G}}}', legend="GPU °C gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEMORY_TEMP{{{G}}}', legend="Mem °C gpu{{gpu}}", refid="B"),
], 0, yy, 8, 7, unit="celsius", fill=0))
gp.append(ts("GPU Clocks", [
    tgt(f'DCGM_FI_DEV_SM_CLOCK{{{G}}}', legend="SM clock gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEM_CLOCK{{{G}}}', legend="Mem clock gpu{{gpu}}", refid="B"),
], 8, yy, 8, 7, unit="megahertz", fill=0))
gp.append(ts("Framebuffer / VRAM (if reported)", [
    tgt(f'DCGM_FI_DEV_FB_USED{{{G}}}*1024*1024', legend="VRAM used gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_FB_FREE{{{G}}}*1024*1024', legend="VRAM free gpu{{gpu}}", refid="B"),
], 16, yy, 8, 7, unit="bytes", mn=0, stack=True)); yy += 7
gp.append(ts("Errors & Throttling (optional fields)", [
    tgt(f'DCGM_FI_DEV_XID_ERRORS{{{G}}}', legend="XID error gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{G}}}', legend="PCIe replay gpu{{gpu}}", refid="B"),
    tgt(f'rate(DCGM_FI_DEV_POWER_VIOLATION{{{G}}}[1m])', legend="power throttle gpu{{gpu}}", refid="C"),
    tgt(f'rate(DCGM_FI_DEV_THERMAL_VIOLATION{{{G}}}[1m])', legend="thermal throttle gpu{{gpu}}", refid="D"),
], 0, yy, 24, 7, unit="short", fill=0)); yy += 7
place("🎮  GPU — NVIDIA Blackwell (DCGM)", gp, collapsed=False)

# =========================================================================
# GRACE CPU & UNIFIED MEMORY
# =========================================================================
cp = []; yy = 0
cp.append(ts("CPU Utilisation by Mode", [tgt(f'avg by(mode)(rate(node_cpu_seconds_total{{{N},mode!="idle"}}[5m]))*100', legend="{{mode}}")],
              0, yy, 12, 7, unit="percent", mn=0, stack=True))
cp.append(ts("Load Average", [
    tgt(f'node_load1{{{N}}}', legend="1m"), tgt(f'node_load5{{{N}}}', legend="5m", refid="B"),
    tgt(f'node_load15{{{N}}}', legend="15m", refid="C"),
], 12, yy, 12, 7, unit="short", mn=0, fill=0)); yy += 7
cp.append(ts("Unified Memory Pool", [
    tgt(f'node_memory_MemTotal_bytes{{{N}}}-node_memory_MemAvailable_bytes{{{N}}}', legend="Used"),
    tgt(f'node_memory_Buffers_bytes{{{N}}}+node_memory_Cached_bytes{{{N}}}', legend="Buffers+Cache", refid="B"),
    tgt(f'node_memory_MemFree_bytes{{{N}}}', legend="Free", refid="C"),
], 0, yy, 16, 8, unit="bytes", mn=0, stack=True))
cp.append(gauge("Memory Pressure", f'100*(1-node_memory_MemAvailable_bytes{{{N}}}/node_memory_MemTotal_bytes{{{N}}})',
                16, yy, 8, 8, unit="percent",
                thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 50}, {"color": "orange", "value": 75}, {"color": "red", "value": 90}])); yy += 8
place("🧠  Grace CPU & Unified Memory", cp, collapsed=False)

# =========================================================================
# vLLM
# =========================================================================
vl = []; yy = 0
vl.append(text("vLLM — Inference Performance", (
    "### 🚀 vLLM   (job `vllm`, `/metrics`)\n"
    "Aggregate throughput is the headline number on a single big GPU. KV-cache near 100% = scheduler under pressure "
    "(lower `--max-num-seqs` or context). Process RAM/CPU come from vLLM's own `process_*` metrics — no extra exporter needed."
), 0, yy, 24, 3)); yy += 3
vl.append(stat("vLLM Server", f'up{{{VL}}}', 0, yy, 4, 5, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
vl.append(model_table("Served Model(s)", [tgt(f'count by(model_name)(vllm:num_requests_running{{{VL}}})', instant=True, fmt="table")],
              4, yy, 8, 5, desc="From the model_name label"))
vl.append(gauge("KV Cache Usage", f'max(vllm:kv_cache_usage_perc{{{VL}}})*100', 12, yy, 4, 5, desc="PagedAttention KV-cache fill (max across models)"))
vl.append(stat("Prefix Cache Hit", SENTJ(f'sum(rate(vllm:prefix_cache_hits_total{{{VL}}}[5m]))/clamp_min(sum(rate(vllm:prefix_cache_queries_total{{{VL}}}[5m])),1)*100', VL),
               16, yy, 4, 5, unit="percent", decimals=1, mappings=OFF("off"),
               thresholds=[{"color": "blue", "value": None}, {"color": "green", "value": 30}]))
vl.append(stat("Process RAM", SENTJ(f'process_resident_memory_bytes{{{VL}}}', VL), 20, yy, 4, 5,
               unit="bytes", mappings=OFF("off"), desc="vLLM server RSS"))
yy += 5
vl.append(ts("Throughput (tok/s)", [
    tgt(f'sum(rate(vllm:generation_tokens_total{{{VL}}}[1m]))', legend="generation"),
    tgt(f'sum(rate(vllm:prompt_tokens_total{{{VL}}}[1m]))', legend="prompt (prefill)", refid="B"),
], 0, yy, 12, 7, unit="none", mn=0, fill=15))
vl.append(ts("Concurrent Requests", [
    tgt(f'sum(vllm:num_requests_running{{{VL}}})', legend="running"),
    tgt(f'sum(vllm:num_requests_waiting{{{VL}}})', legend="waiting (queue)", refid="B"),
], 12, yy, 12, 7, unit="none", mn=0)); yy += 7
vl.append(ts("Time To First Token (s)", [
    tgt(f'histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket{{{VL}}}[1m])) by (le))', legend="TTFT p95"),
    tgt(f'histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket{{{VL}}}[1m])) by (le))', legend="TTFT p50", refid="B"),
], 0, yy, 8, 7, unit="s", mn=0, fill=8))
vl.append(ts("E2E Request Latency (s)", [
    tgt(f'histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{VL}}}[1m])) by (le))', legend="E2E p95"),
    tgt(f'histogram_quantile(0.50, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{VL}}}[1m])) by (le))', legend="E2E p50", refid="B"),
], 8, yy, 8, 7, unit="s", mn=0, fill=8))
vl.append(ts("vLLM Process — CPU & RAM", [
    tgt(f'rate(process_cpu_seconds_total{{{VL}}}[5m])', legend="CPU cores"),
    tgt(f'process_resident_memory_bytes{{{VL}}}', legend="RSS bytes", refid="B"),
], 16, yy, 8, 7, unit="short", fill=8)); yy += 7
place("🚀  vLLM — Inference Performance", vl, collapsed=False)

# =========================================================================
# llama.cpp
# =========================================================================
lc = []; yy = 0
lc.append(text("llama.cpp — Inference Performance", (
    "### 🦙 llama.cpp   (job `llamacpp`, `llama-server --metrics`)\n"
    "Start `llama-server` with `--metrics` and point Prometheus at it. llama.cpp serves one model per process "
    "(no model-name label); container CPU/RAM is optional via cAdvisor."
), 0, yy, 24, 3)); yy += 3
lc.append(stat("llama.cpp Server", f'up{{{LC}}}', 0, yy, 4, 5, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
lc.append(gauge("KV Cache Usage", f'llamacpp:kv_cache_usage_ratio{{{LC}}}*100', 4, yy, 5, 5, desc="KV-cache fill ratio"))
lc.append(stat("Requests Processing", SENTJ(f'llamacpp:requests_processing{{{LC}}}', LC), 9, yy, 5, 5, mappings=OFF("not running")))
lc.append(stat("Requests Deferred", SENTJ(f'llamacpp:requests_deferred{{{LC}}}', LC), 14, yy, 5, 5,
               thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 1}], mappings=OFF("not running")))
lc.append(stat("Container RAM", SENT(f'container_memory_working_set_bytes{{{I},name=~".*llama.*"}}'), 19, yy, 5, 5,
               unit="bytes", mappings=OFF("n/a"), desc="Optional: requires cAdvisor + container named *llama*"))
yy += 5
lc.append(ts("Throughput (tok/s)", [
    tgt(f'llamacpp:predicted_tokens_seconds{{{LC}}}', legend="generation"),
    tgt(f'llamacpp:prompt_tokens_seconds{{{LC}}}', legend="prompt (prefill)", refid="B"),
], 0, yy, 12, 7, unit="none", mn=0, fill=15))
lc.append(ts("KV Cache & Active Slots", [
    tgt(f'llamacpp:kv_cache_usage_ratio{{{LC}}}*100', legend="KV cache %"),
    tgt(f'llamacpp:requests_processing{{{LC}}}', legend="processing", refid="B"),
], 12, yy, 12, 7, unit="short", mn=0, fill=10)); yy += 7
place("🦙  llama.cpp — Inference Performance", lc, collapsed=False)

# =========================================================================
# ollama (optional)
# =========================================================================
ol = []; yy = 0
ol.append(text("ollama — Models & Memory", (
    "### 🤖 ollama   (job `ollama`, optional)\n"
    "Requires an ollama Prometheus exporter (metric names vary by exporter). "
    "Shows loaded-model count and per-model memory footprint where available."
), 0, yy, 24, 3)); yy += 3
ol.append(stat("ollama", f'up{{{OL}}}', 0, yy, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ol.append(stat("Models Loaded", SENTJ(f'ollama_loaded_models{{{OL}}}', OL), 4, yy, 4, 4, mappings=OFF("off")))
ol.append(model_table("Models & RAM (MB)", [tgt(f'ollama_model_ram_mb{{{OL}}}', instant=True, fmt="table")],
              8, yy, 16, 4, desc="Per-model RAM footprint (if the exporter provides it)")); yy += 4
place("🤖  ollama — Models & Memory", ol, collapsed=True)

# =========================================================================
# STORAGE & NETWORK
# =========================================================================
sn = []; yy = 0
sn.append(ts("NVMe Disk I/O", [
    tgt(f'rate(node_disk_read_bytes_total{{{N},device=~"nvme.*|sd.*"}}[5m])', legend="read {{device}}"),
    tgt(f'rate(node_disk_written_bytes_total{{{N},device=~"nvme.*|sd.*"}}[5m])', legend="write {{device}}", refid="B"),
], 0, yy, 12, 7, unit="Bps", fill=8))
sn.append(ts("Filesystem Used %", [
    tgt(f'100*(1-node_filesystem_avail_bytes{{{N},fstype!~"tmpfs|overlay|squashfs"}}/node_filesystem_size_bytes{{{N},fstype!~"tmpfs|overlay|squashfs"}})', legend="{{mountpoint}}")],
    12, yy, 12, 7, unit="percent", mn=0, mx=100, fill=5)); yy += 7
sn.append(ts("Network Throughput", [
    tgt(f'rate(node_network_receive_bytes_total{{{N},{NETFLT}}}[5m])*8', legend="rx {{device}}"),
    tgt(f'rate(node_network_transmit_bytes_total{{{N},{NETFLT}}}[5m])*8', legend="tx {{device}}", refid="B"),
], 0, yy, 12, 7, unit="bps", fill=8))
sn.append(ts("Network Errors & Drops", [
    tgt(f'rate(node_network_receive_errs_total{{{N},{NETFLT}}}[5m])', legend="rx err {{device}}"),
    tgt(f'rate(node_network_transmit_errs_total{{{N},{NETFLT}}}[5m])', legend="tx err {{device}}", refid="B"),
    tgt(f'rate(node_network_receive_drop_total{{{N},{NETFLT}}}[5m])', legend="rx drop {{device}}", refid="C"),
], 12, yy, 12, 7, unit="pps", fill=0)); yy += 7
place("💾  Storage & 🌐 Network", sn, collapsed=True)

# =========================================================================
# templating + dashboard
# =========================================================================
templating = {"list": [
    {"name": "datasource", "type": "datasource", "label": "Prometheus",
     "query": "prometheus", "current": {}, "hide": 0, "refresh": 1,
     "regex": "", "includeAll": False, "multi": False},
    {"name": "instance", "type": "query", "label": "DGX host",
     "datasource": {"type": "prometheus", "uid": "${datasource}"},
     "definition": "label_values(node_uname_info, instance)",
     "query": {"query": "label_values(node_uname_info, instance)", "refId": "StandardVariableQuery"},
     "current": {}, "hide": 0, "refresh": 2, "sort": 1, "includeAll": False, "multi": False},
]}

dashboard = {
    "uid": "dgx-spark-gb10",
    "title": "DGX Spark — GB10 Observability",
    "description": "Full-stack observability for NVIDIA DGX Spark / GB10 Grace-Blackwell: GPU (DCGM), Grace CPU, unified memory, thermal/power, plus vLLM, llama.cpp and ollama inference performance. Portable via $datasource/$instance template variables; offline-aware tiles.",
    "tags": ["dgx-spark", "gb10", "nvidia", "dcgm", "vllm", "llama.cpp", "ollama", "llm"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "editable": True,
    "graphTooltip": 1, "refresh": "10s", "time": {"from": "now-30m", "to": "now"},
    "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "1h"]},
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
        "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts", "type": "dashboard"}]},
    "templating": templating, "panels": panels,
}

import os
os.makedirs("dashboards", exist_ok=True)
with open("dashboards/dgx-spark.json", "w") as f:
    json.dump(dashboard, f, indent=2)
rows = sum(1 for p in panels if p["type"] == "row")
content = sum(1 for p in panels if p["type"] != "row") + sum(len(p.get("panels", [])) for p in panels if p["type"] == "row")
print(f"rows:{rows} content-panels:{content} max-id:{_id[0]}")
print("written dashboards/dgx-spark.json")
