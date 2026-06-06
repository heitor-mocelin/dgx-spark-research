#!/usr/bin/env python3
"""Generate the community 'DGX Spark — GB10 Observability' Grafana dashboard.

Two flavors from the SAME panel definitions (env DASH_FLAVOR):
  generic (default) -> portable: $datasource + $instance template vars, standard job names
                       (node/dcgm/cadvisor/vllm/llamacpp/ollama). This is what ships to GitHub.
  lab               -> bound to the author's homelab labels (fixed datasource, instance="gx10",
                       jobs gx10-node/ai-compute-blackwell/…) so it can be previewed locally with
                       real data before publishing the generic version.

Design: offline-aware tiles (up-status mappings, down-scoped -1 sentinels, friendly No-data text),
threshold lines, and a deep counter set across GPU / Grace CPU / unified memory / inference / IO.
"""
import json, os

FLAVOR = os.environ.get("DASH_FLAVOR", "generic")

if FLAVOR == "lab":
    DS = {"type": "prometheus", "uid": "bfo2i50d29z40c"}
    INST_VAR = False
    NODE = 'instance="gx10",job="gx10-node"'; NJOB = "gx10-node"
    DCGM = 'instance="gx10"';                 DJOB = "ai-compute-blackwell"
    CADV = 'instance="gx10"';                 CJOB = "gx10-cadvisor"
    VLLM = 'job="vllm"';                       VJOB = "vllm"
    LLAMA = 'job="gx10-llamacpp"';             LJOB = "gx10-llamacpp"
    OLLAMA = 'job="gx10-ollama"';              OJOB = "gx10-ollama"
    UID = "dgx-spark-gb10-lab"; TITLE = "DGX Spark — GB10 Observability (lab preview)"
else:
    DS = {"type": "prometheus", "uid": "${datasource}"}
    INST_VAR = True
    NODE = 'job="node",instance="$instance"'; NJOB = "node"
    DCGM = 'job="dcgm",instance="$instance"'; DJOB = "dcgm"
    CADV = 'instance="$instance"';            CJOB = "cadvisor"
    VLLM = 'job="vllm",instance="$instance"';     VJOB = "vllm"
    LLAMA = 'job="llamacpp",instance="$instance"'; LJOB = "llamacpp"
    OLLAMA = 'job="ollama",instance="$instance"';  OJOB = "ollama"
    UID = "dgx-spark-gb10"; TITLE = "DGX Spark — GB10 Observability"

NETFLT = 'device!~"lo|veth.*|docker.*|br-.*|cni.*|flannel.*|cali.*|tap.*|fwbr.*|fwln.*|fwpr.*|vmbr.*"'

_id = [0]
def nid():
    _id[0] += 1
    return _id[0]

T_PCT = [{"color": "green", "value": None}, {"color": "yellow", "value": 70}, {"color": "red", "value": 90}]
T_TEMP = [{"color": "green", "value": None}, {"color": "yellow", "value": 65}, {"color": "red", "value": 85}]
T_KV = [{"color": "green", "value": None}, {"color": "yellow", "value": 80}, {"color": "red", "value": 95}]
T_PWR = [{"color": "blue", "value": None}, {"color": "green", "value": 10}, {"color": "yellow", "value": 60}, {"color": "red", "value": 110}]
MAP_UP = [{"type": "value", "options": {
    "0": {"text": "🔴 Offline", "color": "red", "index": 0},
    "1": {"text": "🟢 Running", "color": "green", "index": 1}}}]
NOVAL = "⚪  No data — exporter / engine not running"

def OFF(text): return [{"type": "value", "options": {"-1": {"text": text, "color": "text", "index": 0}}}]
def SENT(expr): return f'({expr}) or vector(-1)'
def SENTJ(expr, job): return f'({expr}) or (vector(-1) and on() (up{{job="{job}"}} == 0))'


def tgt(expr, legend="", instant=False, refid="A", fmt=None):
    t = {"datasource": DS, "editorMode": "code", "expr": expr, "legendFormat": legend or "__auto",
         "range": not instant, "instant": instant, "refId": refid}
    if fmt: t["format"] = fmt
    return t


def ts(title, targets, x, y, w, h, unit="percent", mn=None, mx=None, desc="", stack=False,
       fill=10, draw="line", tline=None, legend=True, noValue=NOVAL):
    fc = {"drawStyle": draw, "lineInterpolation": "smooth", "lineWidth": 2, "fillOpacity": fill,
          "gradientMode": "opacity", "spanNulls": False, "showPoints": "never", "pointSize": 5,
          "stacking": {"mode": "normal" if stack else "none", "group": "A"},
          "axisCenteredZero": False, "axisColorMode": "text", "axisLabel": "", "axisPlacement": "auto",
          "barAlignment": 0, "thresholdsStyle": {"mode": "line" if tline else "off"}}
    steps = [{"color": "green", "value": None}] + ([{"color": c, "value": v} for v, c in tline] if tline else [])
    fd = {"color": {"mode": "palette-classic"}, "custom": fc, "unit": unit,
          "thresholds": {"mode": "absolute", "steps": steps}}
    if mn is not None: fd["min"] = mn
    if mx is not None: fd["max"] = mx
    if noValue is not None: fd["noValue"] = noValue
    leg = {"displayMode": "table" if legend else "list", "placement": "bottom",
           "calcs": ["lastNotNull", "max", "mean"] if legend else [], "showLegend": bool(legend)}
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "fieldConfig": {"defaults": fd, "overrides": []},
            "options": {"legend": leg, "tooltip": {"mode": "multi", "sort": "desc"}}}


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
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "options": {"mode": "markdown", "content": content}}


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
panels = []; gy = [0]
def place(title, plist, collapsed=False):
    if collapsed:
        panels.append({"id": nid(), "type": "row", "title": title, "collapsed": True,
                       "gridPos": {"x": 0, "y": gy[0], "w": 24, "h": 1}, "panels": plist}); gy[0] += 1
    else:
        panels.append({"id": nid(), "type": "row", "title": title, "collapsed": False,
                       "gridPos": {"x": 0, "y": gy[0], "w": 24, "h": 1}, "panels": []})
        base = gy[0] + 1
        sect_h = max((p["gridPos"]["y"] + p["gridPos"]["h"]) for p in plist) if plist else 0
        for p in plist: p["gridPos"]["y"] += base
        panels.extend(plist); gy[0] = base + sect_h

PWR = f'clamp_min(sum(DCGM_FI_DEV_POWER_USAGE{{{DCGM}}}),1)'   # label-free denominator for tok/W

# =========================================================================
# OVERVIEW & HEALTH
# =========================================================================
ov = []
ov.append(stat("node_exporter", f'up{{{NODE}}}', 0, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("DCGM GPU", f'up{{job="{DJOB}"}}', 4, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("vLLM", f'up{{{VLLM}}}', 8, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("llama.cpp", f'up{{job="{LJOB}"}}', 12, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("ollama", f'up{{job="{OJOB}"}}', 16, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ov.append(stat("cAdvisor", f'up{{job="{CJOB}"}}', 20, 0, 4, 3, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
# KPI row 1
ov.append(stat("GPU Util", SENTJ(f'avg(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}})', DJOB), 0, 3, 4, 4, unit="percent", thresholds=T_PCT, mappings=OFF("GPU off")))
ov.append(stat("GPU Temp", SENTJ(f'max(DCGM_FI_DEV_GPU_TEMP{{{DCGM}}})', DJOB), 4, 3, 4, 4, unit="celsius", thresholds=T_TEMP, mappings=OFF("GPU off")))
ov.append(stat("GPU Power", SENTJ(f'sum(DCGM_FI_DEV_POWER_USAGE{{{DCGM}}})', DJOB), 8, 3, 4, 4, unit="watt", thresholds=T_PWR, mappings=OFF("GPU off")))
ov.append(stat("Unified Mem", SENTJ(f'100*(1-node_memory_MemAvailable_bytes{{{NODE}}}/node_memory_MemTotal_bytes{{{NODE}}})', NJOB), 12, 3, 4, 4, unit="percent", thresholds=T_PCT, decimals=1, mappings=OFF("node off")))
ov.append(stat("CPU", SENTJ(f'100-(avg(rate(node_cpu_seconds_total{{{NODE},mode="idle"}}[5m]))*100)', NJOB), 16, 3, 4, 4, unit="percent", thresholds=T_PCT, decimals=1, mappings=OFF("node off")))
ov.append(stat("Uptime", SENTJ(f'node_time_seconds{{{NODE}}}-node_boot_time_seconds{{{NODE}}}', NJOB), 20, 3, 4, 4, unit="s", decimals=0, color_mode="none", mappings=OFF("off")))
# KPI row 2 (LLM-centric)
ov.append(stat("vLLM tok/s", SENTJ(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))', VJOB), 0, 7, 4, 4, unit="none", decimals=0, color_mode="none", mappings=OFF("vLLM off")))
ov.append(stat("Tokens / Watt", SENTJ(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))/{PWR}', VJOB), 4, 7, 4, 4, unit="none", decimals=2, mappings=OFF("vLLM off"), desc="Generation tokens per Watt of GPU power"))
ov.append(stat("vLLM KV Cache", SENTJ(f'max(vllm:kv_cache_usage_perc{{{VLLM}}})*100', VJOB), 8, 7, 4, 4, unit="percent", decimals=1, thresholds=T_KV, mappings=OFF("vLLM off")))
ov.append(stat("vLLM Queue", SENTJ(f'sum(vllm:num_requests_waiting{{{VLLM}}})', VJOB), 12, 7, 4, 4, unit="none", thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 1}], mappings=OFF("vLLM off")))
ov.append(stat("llama.cpp tok/s", SENTJ(f'sum(llamacpp:predicted_tokens_seconds{{{LLAMA}}})', LJOB), 16, 7, 4, 4, unit="none", decimals=0, color_mode="none", mappings=OFF("not running")))
ov.append(stat("Load (1m)", SENTJ(f'node_load1{{{NODE}}}', NJOB), 20, 7, 4, 4, unit="none", decimals=2, mappings=OFF("off")))
place("📊  Overview & Health", ov, collapsed=False)

# =========================================================================
# GPU — COMPUTE & ACTIVITY (DCGM)
# =========================================================================
gc = []; yy = 0
gc.append(text("GPU Compute & Activity", (
    "### 🎮 GPU — Compute & Activity (DCGM)\n"
    "Utilisation plus **DCP profiling** engine-activity ratios (SM/Tensor/FP/DRAM active). Profiling metrics "
    "(`DCGM_FI_PROF_*`) require a DCGM exporter built with the profiling field set — they show *No data* if your "
    "exporter omits them."), 0, yy, 24, 3)); yy += 3
gc.append(ts("GPU Utilisation & Mem-Copy", [
    tgt(f'DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}', legend="SM util gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEM_COPY_UTIL{{{DCGM}}}', legend="mem-copy gpu{{gpu}}", refid="B"),
], 0, yy, 12, 7, unit="percent", mn=0, mx=100, tline=[(90, "red")]))
gc.append(ts("Engine Activity (profiling)", [
    tgt(f'DCGM_FI_PROF_GR_ENGINE_ACTIVE{{{DCGM}}}*100', legend="graphics gpu{{gpu}}"),
    tgt(f'DCGM_FI_PROF_SM_ACTIVE{{{DCGM}}}*100', legend="SM active gpu{{gpu}}", refid="B"),
    tgt(f'DCGM_FI_PROF_SM_OCCUPANCY{{{DCGM}}}*100', legend="SM occupancy gpu{{gpu}}", refid="C"),
], 12, yy, 12, 7, unit="percent", mn=0, mx=100, fill=8)); yy += 7
gc.append(ts("Pipe Activity — Tensor / FP", [
    tgt(f'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{DCGM}}}*100', legend="tensor gpu{{gpu}}"),
    tgt(f'DCGM_FI_PROF_PIPE_FP16_ACTIVE{{{DCGM}}}*100', legend="fp16 gpu{{gpu}}", refid="B"),
    tgt(f'DCGM_FI_PROF_PIPE_FP32_ACTIVE{{{DCGM}}}*100', legend="fp32 gpu{{gpu}}", refid="C"),
    tgt(f'DCGM_FI_PROF_PIPE_FP64_ACTIVE{{{DCGM}}}*100', legend="fp64 gpu{{gpu}}", refid="D"),
], 0, yy, 12, 7, unit="percent", mn=0, mx=100, fill=6))
gc.append(ts("DRAM Activity & Encoder/Decoder", [
    tgt(f'DCGM_FI_PROF_DRAM_ACTIVE{{{DCGM}}}*100', legend="DRAM active gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_ENC_UTIL{{{DCGM}}}', legend="encoder gpu{{gpu}}", refid="B"),
    tgt(f'DCGM_FI_DEV_DEC_UTIL{{{DCGM}}}', legend="decoder gpu{{gpu}}", refid="C"),
], 12, yy, 12, 7, unit="percent", mn=0, mx=100, fill=6)); yy += 7
place("🎮  GPU — Compute & Activity (DCGM)", gc, collapsed=False)

# =========================================================================
# GPU — MEMORY, POWER & THERMAL
# =========================================================================
gm = []; yy = 0
gm.append(ts("Framebuffer / VRAM (if reported)", [
    tgt(f'DCGM_FI_DEV_FB_USED{{{DCGM}}}*1024*1024', legend="used gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_FB_FREE{{{DCGM}}}*1024*1024', legend="free gpu{{gpu}}", refid="B"),
], 0, yy, 8, 7, unit="bytes", mn=0, stack=True,
    desc="GB10 unified memory may not expose framebuffer — see Grace CPU & Unified Memory"))
gm.append(ts("Power & Limit", [
    tgt(f'DCGM_FI_DEV_POWER_USAGE{{{DCGM}}}', legend="power W gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_POWER_MGMT_LIMIT{{{DCGM}}}', legend="limit W gpu{{gpu}}", refid="B"),
    tgt(f'rate(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{{DCGM}}}[1m])/1000', legend="energy rate W gpu{{gpu}}", refid="C"),
], 8, yy, 8, 7, unit="watt", fill=8))
gm.append(ts("Temperature", [
    tgt(f'DCGM_FI_DEV_GPU_TEMP{{{DCGM}}}', legend="GPU °C gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEMORY_TEMP{{{DCGM}}}', legend="Mem °C gpu{{gpu}}", refid="B"),
], 16, yy, 8, 7, unit="celsius", fill=0, tline=[(85, "red")])); yy += 7
gm.append(ts("Clocks", [
    tgt(f'DCGM_FI_DEV_SM_CLOCK{{{DCGM}}}', legend="SM clock gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_MEM_CLOCK{{{DCGM}}}', legend="Mem clock gpu{{gpu}}", refid="B"),
], 0, yy, 12, 7, unit="megahertz", fill=0))
gm.append(ts("Power Efficiency (tokens / Watt)", [
    tgt(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))/{PWR}', legend="vLLM tok/W")],
    12, yy, 12, 7, unit="none", mn=0, fill=10, desc="Higher = more tokens per watt of GPU power")); yy += 7
place("🔋  GPU — Memory, Power & Thermal", gm, collapsed=False)

# =========================================================================
# GPU — INTERCONNECT & RELIABILITY
# =========================================================================
gi = []; yy = 0
gi.append(ts("PCIe Throughput (profiling)", [
    tgt(f'DCGM_FI_PROF_PCIE_TX_BYTES{{{DCGM}}}', legend="PCIe TX gpu{{gpu}}"),
    tgt(f'DCGM_FI_PROF_PCIE_RX_BYTES{{{DCGM}}}', legend="PCIe RX gpu{{gpu}}", refid="B"),
], 0, yy, 8, 7, unit="Bps", fill=8))
gi.append(ts("NVLink Throughput (profiling)", [
    tgt(f'DCGM_FI_PROF_NVLINK_TX_BYTES{{{DCGM}}}', legend="NVLink TX gpu{{gpu}}"),
    tgt(f'DCGM_FI_PROF_NVLINK_RX_BYTES{{{DCGM}}}', legend="NVLink RX gpu{{gpu}}", refid="B"),
], 8, yy, 8, 7, unit="Bps", fill=8))
gi.append(ts("Errors & PCIe Replays", [
    tgt(f'DCGM_FI_DEV_XID_ERRORS{{{DCGM}}}', legend="XID gpu{{gpu}}"),
    tgt(f'DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{DCGM}}}', legend="PCIe replay gpu{{gpu}}", refid="B"),
], 16, yy, 8, 7, unit="short", fill=0)); yy += 7
gi.append(ts("Throttle Violations (ns/s)", [
    tgt(f'rate(DCGM_FI_DEV_POWER_VIOLATION{{{DCGM}}}[1m])', legend="power gpu{{gpu}}"),
    tgt(f'rate(DCGM_FI_DEV_THERMAL_VIOLATION{{{DCGM}}}[1m])', legend="thermal gpu{{gpu}}", refid="B"),
    tgt(f'rate(DCGM_FI_DEV_BOARD_LIMIT_VIOLATION{{{DCGM}}}[1m])', legend="board gpu{{gpu}}", refid="C"),
    tgt(f'rate(DCGM_FI_DEV_RELIABILITY_VIOLATION{{{DCGM}}}[1m])', legend="reliability gpu{{gpu}}", refid="D"),
], 0, yy, 24, 6, unit="short", fill=0)); yy += 6
place("🔌  GPU — Interconnect & Reliability", gi, collapsed=True)

# =========================================================================
# GRACE CPU
# =========================================================================
cp = []; yy = 0
cp.append(ts("CPU Utilisation by Mode", [tgt(f'avg by(mode)(rate(node_cpu_seconds_total{{{NODE},mode!="idle"}}[5m]))*100', legend="{{mode}}")],
              0, yy, 12, 7, unit="percent", mn=0, stack=True))
cp.append(ts("Per-Core Utilisation", [tgt(f'100-(rate(node_cpu_seconds_total{{{NODE},mode="idle"}}[5m])*100)', legend="cpu{{cpu}}")],
              12, yy, 12, 7, unit="percent", mn=0, mx=100, fill=4, legend=False, desc="One line per logical core")); yy += 7
cp.append(ts("CPU Frequency", [tgt(f'avg(node_cpu_scaling_frequency_hertz{{{NODE}}})', legend="avg"),
              tgt(f'max(node_cpu_scaling_frequency_hertz{{{NODE}}})', legend="max", refid="B")],
              0, yy, 8, 7, unit="hertz", fill=4))
cp.append(ts("Load Average", [tgt(f'node_load1{{{NODE}}}', legend="1m"),
              tgt(f'node_load5{{{NODE}}}', legend="5m", refid="B"), tgt(f'node_load15{{{NODE}}}', legend="15m", refid="C")],
              8, yy, 8, 7, unit="short", mn=0, fill=0))
cp.append(ts("Context Switches & Interrupts (/s)", [
    tgt(f'rate(node_context_switches_total{{{NODE}}}[5m])', legend="ctx switches"),
    tgt(f'rate(node_intr_total{{{NODE}}}[5m])', legend="interrupts", refid="B")],
    16, yy, 8, 7, unit="short", fill=4)); yy += 7
cp.append(ts("Processes", [tgt(f'node_procs_running{{{NODE}}}', legend="running"),
              tgt(f'node_procs_blocked{{{NODE}}}', legend="blocked", refid="B")],
              0, yy, 12, 6, unit="short", mn=0, fill=4))
cp.append(ts("CPU Pressure Stall (PSI)", [
    tgt(f'rate(node_pressure_cpu_waiting_seconds_total{{{NODE}}}[5m])*100', legend="some stalled %")],
    12, yy, 12, 6, unit="percent", mn=0, fill=8, desc="% of time runnable tasks waited on CPU")); yy += 6
place("🧠  Grace CPU (20-core ARM)", cp, collapsed=False)

# =========================================================================
# UNIFIED MEMORY
# =========================================================================
um = []; yy = 0
um.append(text("Unified Memory", (
    "### 💠 Unified Memory\n"
    "On GB10 the CPU and GPU **share one LPDDR5 pool** — this *is* the GPU memory. Watch the pressure gauge and "
    "swap: under memory pressure the inference engines are the first to suffer."), 0, yy, 24, 2)); yy += 2
um.append(ts("Memory Pool", [
    tgt(f'node_memory_MemTotal_bytes{{{NODE}}}-node_memory_MemAvailable_bytes{{{NODE}}}', legend="Used"),
    tgt(f'node_memory_Buffers_bytes{{{NODE}}}+node_memory_Cached_bytes{{{NODE}}}', legend="Buffers+Cache", refid="B"),
    tgt(f'node_memory_MemFree_bytes{{{NODE}}}', legend="Free", refid="C"),
], 0, yy, 12, 7, unit="bytes", mn=0, stack=True))
um.append(gauge("Memory Pressure", f'100*(1-node_memory_MemAvailable_bytes{{{NODE}}}/node_memory_MemTotal_bytes{{{NODE}}})',
                12, yy, 6, 7, unit="percent",
                thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 50}, {"color": "orange", "value": 75}, {"color": "red", "value": 90}]))
um.append(ts("Swap", [tgt(f'node_memory_SwapTotal_bytes{{{NODE}}}-node_memory_SwapFree_bytes{{{NODE}}}', legend="swap used"),
              tgt(f'node_memory_SwapTotal_bytes{{{NODE}}}', legend="swap total", refid="B")],
              18, yy, 6, 7, unit="bytes", mn=0, fill=6)); yy += 7
um.append(ts("Memory Breakdown", [
    tgt(f'node_memory_Slab_bytes{{{NODE}}}', legend="slab"),
    tgt(f'node_memory_PageTables_bytes{{{NODE}}}', legend="page tables", refid="B"),
    tgt(f'node_memory_Dirty_bytes{{{NODE}}}', legend="dirty", refid="C"),
    tgt(f'node_memory_Mapped_bytes{{{NODE}}}', legend="mapped", refid="D"),
    tgt(f'node_memory_AnonPages_bytes{{{NODE}}}', legend="anon", refid="E"),
], 0, yy, 12, 6, unit="bytes", mn=0, fill=4))
um.append(ts("Memory Pressure (PSI) & OOM Kills", [
    tgt(f'rate(node_pressure_memory_waiting_seconds_total{{{NODE}}}[5m])*100', legend="mem stall %"),
    tgt(f'rate(node_vmstat_oom_kill{{{NODE}}}[5m])*60', legend="OOM kills/min", refid="B"),
], 12, yy, 12, 6, unit="short", mn=0, fill=8)); yy += 6
place("💠  Unified Memory (128 GB LPDDR5)", um, collapsed=False)

# =========================================================================
# vLLM
# =========================================================================
vl = []; yy = 0
vl.append(text("vLLM — Inference Performance", (
    "### 🚀 vLLM   (job `vllm`)\n"
    "Aggregate throughput, KV-cache pressure, latency percentiles, and request shape. Process RAM/CPU from vLLM's "
    "own `process_*` metrics — no extra exporter needed."), 0, yy, 24, 3)); yy += 3
vl.append(stat("vLLM Server", f'up{{{VLLM}}}', 0, yy, 4, 5, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
vl.append(model_table("Served Model(s)", [tgt(f'count by(model_name)(vllm:num_requests_running{{{VLLM}}})', instant=True, fmt="table")], 4, yy, 8, 5, desc="From model_name label"))
vl.append(gauge("KV Cache Usage", f'max(vllm:kv_cache_usage_perc{{{VLLM}}})*100', 12, yy, 4, 5, desc="PagedAttention KV-cache fill"))
vl.append(stat("Prefix Cache Hit", SENTJ(f'sum(rate(vllm:prefix_cache_hits_total{{{VLLM}}}[5m]))/clamp_min(sum(rate(vllm:prefix_cache_queries_total{{{VLLM}}}[5m])),1)*100', VJOB), 16, yy, 4, 5, unit="percent", decimals=1, mappings=OFF("off"), thresholds=[{"color": "blue", "value": None}, {"color": "green", "value": 30}]))
vl.append(stat("Process RAM", SENTJ(f'process_resident_memory_bytes{{{VLLM}}}', VJOB), 20, yy, 4, 5, unit="bytes", mappings=OFF("off"), desc="vLLM server RSS"))
yy += 5
vl.append(ts("Throughput (tok/s)", [
    tgt(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))', legend="generation"),
    tgt(f'sum(rate(vllm:prompt_tokens_total{{{VLLM}}}[1m]))', legend="prompt (prefill)", refid="B"),
], 0, yy, 12, 7, unit="none", mn=0, fill=15))
vl.append(ts("Concurrent Requests", [
    tgt(f'sum(vllm:num_requests_running{{{VLLM}}})', legend="running"),
    tgt(f'sum(vllm:num_requests_waiting{{{VLLM}}})', legend="waiting (queue)", refid="B"),
], 12, yy, 12, 7, unit="none", mn=0)); yy += 7
vl.append(ts("Time To First Token (s)", [
    tgt(f'histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket{{{VLLM}}}[1m])) by (le))', legend="p95"),
    tgt(f'histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket{{{VLLM}}}[1m])) by (le))', legend="p50", refid="B"),
], 0, yy, 8, 7, unit="s", mn=0, fill=8))
vl.append(ts("E2E Request Latency (s)", [
    tgt(f'histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{VLLM}}}[1m])) by (le))', legend="p95"),
    tgt(f'histogram_quantile(0.50, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{VLLM}}}[1m])) by (le))', legend="p50", refid="B"),
], 8, yy, 8, 7, unit="s", mn=0, fill=8))
vl.append(ts("KV Cache Usage %", [tgt(f'vllm:kv_cache_usage_perc{{{VLLM}}}*100', legend="{{model_name}}")],
              16, yy, 8, 7, unit="percent", mn=0, mx=100, fill=12, tline=[(95, "red")])); yy += 7
place("🚀  vLLM — Inference Performance", vl, collapsed=False)

# =========================================================================
# vLLM ADVANCED
# =========================================================================
va = []; yy = 0
va.append(stat("Tokens / Watt", SENTJ(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))/{PWR}', VJOB), 0, yy, 5, 4, unit="none", decimals=2, mappings=OFF("off"), desc="Energy efficiency"))
va.append(stat("Request Success /s", SENTJ(f'sum(rate(vllm:request_success_total{{{VLLM}}}[5m]))', VJOB), 5, yy, 5, 4, unit="reqps", decimals=2, mappings=OFF("off")))
va.append(stat("Preemptions /min", SENTJ(f'sum(rate(vllm:num_preemptions_total{{{VLLM}}}[5m]))*60', VJOB), 10, yy, 5, 4, unit="none", decimals=1, mappings=OFF("off"), thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 1}, {"color": "red", "value": 10}], desc="KV-pressure evictions; should be ~0"))
va.append(stat("Prompt-Cache Eff.", SENTJ(f'sum(rate(vllm:prompt_tokens_cached_total{{{VLLM}}}[5m]))/clamp_min(sum(rate(vllm:prompt_tokens_total{{{VLLM}}}[5m])),1)*100', VJOB), 15, yy, 5, 4, unit="percent", decimals=1, mappings=OFF("off"), thresholds=[{"color": "blue", "value": None}, {"color": "green", "value": 30}]))
va.append(stat("Avg Batch (iter tokens)", SENTJ(f'sum(rate(vllm:iteration_tokens_total_sum{{{VLLM}}}[1m]))/clamp_min(sum(rate(vllm:iteration_tokens_total_count{{{VLLM}}}[1m])),1)', VJOB), 20, yy, 4, 4, unit="none", decimals=0, mappings=OFF("off"), desc="Mean tokens processed per engine step"))
yy += 4
va.append(ts("Energy Efficiency (tokens / Watt)", [tgt(f'sum(rate(vllm:generation_tokens_total{{{VLLM}}}[1m]))/{PWR}', legend="tok/W")], 0, yy, 8, 7, unit="none", mn=0, fill=12))
va.append(ts("Scheduler — Preemptions & Success", [
    tgt(f'sum(rate(vllm:num_preemptions_total{{{VLLM}}}[5m]))', legend="preemptions/s"),
    tgt(f'sum(rate(vllm:request_success_total{{{VLLM}}}[5m]))', legend="success/s", refid="B"),
], 8, yy, 8, 7, unit="short", mn=0, fill=8))
va.append(ts("Avg Request Shape (tokens)", [
    tgt(f'sum(rate(vllm:request_prompt_tokens_sum{{{VLLM}}}[5m]))/clamp_min(sum(rate(vllm:request_prompt_tokens_count{{{VLLM}}}[5m])),1)', legend="avg prompt"),
    tgt(f'sum(rate(vllm:request_generation_tokens_sum{{{VLLM}}}[5m]))/clamp_min(sum(rate(vllm:request_generation_tokens_count{{{VLLM}}}[5m])),1)', legend="avg generation", refid="B"),
], 16, yy, 8, 7, unit="none", mn=0, fill=6)); yy += 7
place("⚡  vLLM — Efficiency & Scheduler Health", va, collapsed=False)

# =========================================================================
# llama.cpp
# =========================================================================
lc = []; yy = 0
lc.append(text("llama.cpp — Inference Performance", (
    "### 🦙 llama.cpp   (job `llamacpp`, `llama-server --metrics`)\n"
    "Forward-ready: panels populate when `llama-server` runs with `--metrics`."), 0, yy, 24, 3)); yy += 3
lc.append(stat("llama.cpp Server", f'up{{job="{LJOB}"}}', 0, yy, 4, 5, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
lc.append(gauge("KV Cache Usage", f'llamacpp:kv_cache_usage_ratio{{{LLAMA}}}*100', 4, yy, 5, 5))
lc.append(stat("Requests Processing", SENTJ(f'llamacpp:requests_processing{{{LLAMA}}}', LJOB), 9, yy, 5, 5, mappings=OFF("not running")))
lc.append(stat("Requests Deferred", SENTJ(f'llamacpp:requests_deferred{{{LLAMA}}}', LJOB), 14, yy, 5, 5, thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 1}], mappings=OFF("not running")))
lc.append(stat("KV Cache Tokens", SENTJ(f'llamacpp:kv_cache_tokens{{{LLAMA}}}', LJOB), 19, yy, 5, 5, unit="none", mappings=OFF("n/a")))
yy += 5
lc.append(ts("Throughput (tok/s)", [
    tgt(f'llamacpp:predicted_tokens_seconds{{{LLAMA}}}', legend="generation"),
    tgt(f'llamacpp:prompt_tokens_seconds{{{LLAMA}}}', legend="prompt (prefill)", refid="B"),
], 0, yy, 12, 7, unit="none", mn=0, fill=15))
lc.append(ts("Cumulative Tokens & Busy Slots", [
    tgt(f'rate(llamacpp:tokens_predicted_total{{{LLAMA}}}[5m])', legend="predicted tok/s (rate)"),
    tgt(f'llamacpp:n_busy_slots_per_decode{{{LLAMA}}}', legend="busy slots/decode", refid="B"),
], 12, yy, 12, 7, unit="short", mn=0, fill=8)); yy += 7
place("🦙  llama.cpp — Inference Performance", lc, collapsed=True)

# =========================================================================
# ollama
# =========================================================================
ol = []; yy = 0
ol.append(stat("ollama", f'up{{job="{OJOB}"}}', 0, yy, 4, 4, mappings=MAP_UP, color_mode="background", noValue="🔴 Offline"))
ol.append(stat("Models Loaded", SENTJ(f'ollama_loaded_models{{{OLLAMA}}}', OJOB), 4, yy, 4, 4, mappings=OFF("off")))
ol.append(model_table("Models & RAM (MB)", [tgt(f'ollama_model_ram_mb{{{OLLAMA}}}', instant=True, fmt="table")], 8, yy, 16, 4)); yy += 4
place("🤖  ollama — Models & Memory", ol, collapsed=True)

# =========================================================================
# STORAGE
# =========================================================================
sg = []; yy = 0
sg.append(ts("Disk Throughput", [
    tgt(f'rate(node_disk_read_bytes_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])', legend="read {{device}}"),
    tgt(f'rate(node_disk_written_bytes_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])', legend="write {{device}}", refid="B"),
], 0, yy, 8, 7, unit="Bps", fill=8))
sg.append(ts("Disk IOPS", [
    tgt(f'rate(node_disk_reads_completed_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])', legend="read {{device}}"),
    tgt(f'rate(node_disk_writes_completed_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])', legend="write {{device}}", refid="B"),
], 8, yy, 8, 7, unit="iops", fill=4))
sg.append(ts("Disk Busy % & Latency", [
    tgt(f'rate(node_disk_io_time_seconds_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])*100', legend="busy% {{device}}"),
    tgt(f'rate(node_disk_io_time_weighted_seconds_total{{{NODE},device=~"nvme.*|sd.*"}}[5m])', legend="await {{device}}", refid="B"),
], 16, yy, 8, 7, unit="percent", mn=0, fill=4)); yy += 7
sg.append(ts("Filesystem Used %", [tgt(f'100*(1-node_filesystem_avail_bytes{{{NODE},fstype!~"tmpfs|overlay|squashfs|ramfs"}}/node_filesystem_size_bytes{{{NODE},fstype!~"tmpfs|overlay|squashfs|ramfs"}})', legend="{{mountpoint}}")],
              0, yy, 8, 7, unit="percent", mn=0, mx=100, fill=5, tline=[(90, "red")]))
sg.append(ts("Inodes Used %", [tgt(f'100*(1-node_filesystem_files_free{{{NODE},fstype!~"tmpfs|overlay|squashfs|ramfs"}}/node_filesystem_files{{{NODE},fstype!~"tmpfs|overlay|squashfs|ramfs"}})', legend="{{mountpoint}}")],
              8, yy, 8, 7, unit="percent", mn=0, mx=100, fill=4))
sg.append(ts("NVMe Temperature", [tgt(f'node_hwmon_temp_celsius{{{NODE},chip=~".*nvme.*"}}', legend="{{chip}}")],
              16, yy, 8, 7, unit="celsius", fill=0, tline=[(75, "red")])); yy += 7
place("💾  Storage", sg, collapsed=True)

# =========================================================================
# NETWORK
# =========================================================================
nw = []; yy = 0
nw.append(ts("Throughput", [
    tgt(f'rate(node_network_receive_bytes_total{{{NODE},{NETFLT}}}[5m])*8', legend="rx {{device}}"),
    tgt(f'rate(node_network_transmit_bytes_total{{{NODE},{NETFLT}}}[5m])*8', legend="tx {{device}}", refid="B"),
], 0, yy, 8, 7, unit="bps", fill=8))
nw.append(ts("Packets /s", [
    tgt(f'rate(node_network_receive_packets_total{{{NODE},{NETFLT}}}[5m])', legend="rx {{device}}"),
    tgt(f'rate(node_network_transmit_packets_total{{{NODE},{NETFLT}}}[5m])', legend="tx {{device}}", refid="B"),
], 8, yy, 8, 7, unit="pps", fill=4))
nw.append(ts("TCP Connections & Retransmits", [
    tgt(f'node_netstat_Tcp_CurrEstab{{{NODE}}}', legend="established"),
    tgt(f'rate(node_netstat_Tcp_RetransSegs{{{NODE}}}[5m])', legend="retrans/s", refid="B"),
], 16, yy, 8, 7, unit="short", mn=0, fill=4)); yy += 7
nw.append(ts("Errors & Drops", [
    tgt(f'rate(node_network_receive_errs_total{{{NODE},{NETFLT}}}[5m])', legend="rx err {{device}}"),
    tgt(f'rate(node_network_transmit_errs_total{{{NODE},{NETFLT}}}[5m])', legend="tx err {{device}}", refid="B"),
    tgt(f'rate(node_network_receive_drop_total{{{NODE},{NETFLT}}}[5m])', legend="rx drop {{device}}", refid="C"),
], 0, yy, 24, 6, unit="pps", fill=0)); yy += 6
place("🌐  Network", nw, collapsed=True)

# =========================================================================
# templating + dashboard
# =========================================================================
tvars = [{"name": "datasource", "type": "datasource", "label": "Prometheus", "query": "prometheus",
          "current": {}, "hide": 0, "refresh": 1, "regex": "", "includeAll": False, "multi": False}]
if INST_VAR:
    tvars.append({"name": "instance", "type": "query", "label": "DGX host",
                  "datasource": {"type": "prometheus", "uid": "${datasource}"},
                  "definition": "label_values(node_uname_info, instance)",
                  "query": {"query": "label_values(node_uname_info, instance)", "refId": "StandardVariableQuery"},
                  "current": {}, "hide": 0, "refresh": 2, "sort": 1, "includeAll": False, "multi": False})

ann_ds = DS if FLAVOR == "lab" else {"type": "prometheus", "uid": "${datasource}"}
annotations = {"list": [
    {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True,
     "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"},
    {"datasource": ann_ds, "enable": True, "hide": False, "iconColor": "orange",
     "name": "Engine restarts / outages",
     "expr": f'changes(up{{job=~"{VJOB}|{LJOB}|{DJOB}"}}[2m]) > 0',
     "titleFormat": "{{job}} state change", "step": "60s"},
]}

dashboard = {
    "uid": UID, "title": TITLE,
    "description": "Full-stack observability for NVIDIA DGX Spark / GB10 Grace-Blackwell: GPU compute/activity, memory/power/thermal, interconnect & reliability (DCGM), Grace CPU, unified memory, plus vLLM / llama.cpp / ollama inference performance incl. tokens-per-Watt efficiency. Offline-aware tiles; portable via $datasource/$instance.",
    "tags": ["dgx-spark", "gb10", "nvidia", "dcgm", "vllm", "llama.cpp", "ollama", "llm"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "editable": True,
    "graphTooltip": 1, "refresh": "10s", "time": {"from": "now-30m", "to": "now"},
    "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "1h"]},
    "annotations": annotations, "templating": {"list": tvars}, "panels": panels,
}

import os as _os
_os.makedirs("dashboards", exist_ok=True)
outfile = "dashboards/dgx-spark.json" if FLAVOR == "generic" else "dashboards/dgx-spark.lab.json"
with open(outfile, "w") as f:
    json.dump(dashboard, f, indent=2)
rows = sum(1 for p in panels if p["type"] == "row")
content = sum(1 for p in panels if p["type"] != "row") + sum(len(p.get("panels", [])) for p in panels if p["type"] == "row")
print(f"flavor:{FLAVOR} uid:{UID} rows:{rows} content-panels:{content} -> {outfile}")
