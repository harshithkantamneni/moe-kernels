#!/usr/bin/env bash
# Descriptive per-kernel split for the v10 blind models (2026-11-21). Sequential, one model at a time.
set -uo pipefail
W=$HOME/code/moe-kernels/.claude/worktrees/agent-a675b7a42302380f5   # frozen v10 code, f1a0c0c4
D=$HOME/moe-kernels-exfil/v10-blind/split/run/vm/results
cd "$(dirname "$0")"
for m in lfm2-8b-a1b phi-mini-moe granite-4.0-h-small; do
  for t in 64x128x64w8s3 128x128x64w8s3; do
    out="$m-${t%%x*}x${t#*x}"; out="$m-$(echo $t | cut -d x -f1).jsonl"
    echo "$(date -u +%T) $m $t -> $out"
    /usr/bin/time -l uv run -q --no-project --python 3.12 --with numpy==2.5.3 --with torch==2.13.0 --with pyyaml==6.0.3 \
      --with pandas==3.0.6 --with pyarrow==25.0.1 python -I split.py "$W" "$D/prod-$m-20261121T073522Z/A1410" \
      "$D/prod-$m-20261121T073522Z/sag/par.json" "$m" "$t" "$out" 150 2>&1 | grep -E "calls,|patched|done|Error|error|maximum resident" 
  done
done
echo ALL-DONE
