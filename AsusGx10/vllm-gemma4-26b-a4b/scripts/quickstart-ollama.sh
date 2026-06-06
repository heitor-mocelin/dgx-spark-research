#!/usr/bin/env bash
#
# quickstart-ollama.sh — the fastest path to Gemma 4 locally (any machine with enough RAM/VRAM).
# Ollama auto-selects a quantization that fits your memory. Needs Ollama v0.20.0+.
#
#   ./quickstart-ollama.sh            # 26B-A4B MoE (best quality-to-memory)
#   MODEL=gemma4:31b ./quickstart-ollama.sh
#
# For reliable tool-calling, prefer the vLLM path (launch-gemma4-vllm.sh) — Ollama's tool-call
# parser had bugs with Gemma 4's hybrid attention at launch (sources/g08).
set -euo pipefail
MODEL="${MODEL:-gemma4:26b-moe}"   # alt: gemma4:31b (max quality), gemma4:4b / gemma4:2b (edge)

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com (need v0.20.0+)"; exit 1; }
ver="$(ollama --version 2>/dev/null || true)"; echo "ollama: $ver"
echo "pulling $MODEL (Ollama picks a fitting quantization automatically)…"
ollama pull "$MODEL"
echo "running $MODEL — Ctrl-D to exit"
exec ollama run "$MODEL"
