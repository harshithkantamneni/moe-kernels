#!/usr/bin/env bash
# Second recovery (Qwen2-57B, 2026-09-28): the deep G=2 page again at the 1710 lock, treads 1 to 9
# (its first deep page failed V0: 206 of 243 cells).
set -uo pipefail
. ~/moe/env.sh && cd "$REPO" || exit 2
export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
S=$SESSION_ROOT; ST=$S/gh200-driver/status; B=$(cat "$S/base-tag.txt")
R3=(--model qwen2-57b-a14b --block-m 32 --repeats 9 --duty 0.25 --seed 0 --declared-copies 9)
log() { echo "$(date -u +%FT%TZ) RECOVERY2 $*" >> "$ST"; }
push() { bash "$REPO/scripts/vm_results_push.sh" push --message "$1" >/dev/null 2>&1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "GPU busy: stop"; exit 2; }
moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null
log "start: deep G=2 at 1710, treads 9 (tag $B-deep2b)"
python3 scripts/locked_r3.py --session-tag "$B-deep2b" --groups 2 --locks 1710 --run-cap-s 3600 -- "${R3[@]}" --treads 9 \
  > "$S/logs/locked_r3-$B-deep2b.log" 2>&1
rc=$?; log "deep G=2 retake locked_r3 exit $rc: $(grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' $S/locked-r3/$B-deep2b/status | cut -c22-150)"
sudo -n nvidia-smi -rgc >/dev/null
log "END"; push "recovery 2: deep G=2 retake exit $rc; END"
