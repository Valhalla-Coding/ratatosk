#!/usr/bin/env bash
# Quick check of a running Ratatosk. Run on Idun. Set "default_model" in Settings first.
# Usage: scripts/smoke_idun.sh [llm_port] [ui_port]     (defaults 11434 and 5960)
llm="${1:-11434}"; ui="${2:-5960}"
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
echo "GPU memory before: $(mem) MB"
echo "--- models"; curl -s -m 30 "localhost:$llm/v1/models" | head -c 400; echo
echo "--- short streamed completion"
curl -sN -m 900 "localhost:$llm/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"Count: 1, 2, 3,","max_tokens":12,"stream":true}' | head -c 600; echo
echo "--- status"; curl -s "localhost:$ui/api/status"; echo
echo "GPU memory after: $(mem) MB"
