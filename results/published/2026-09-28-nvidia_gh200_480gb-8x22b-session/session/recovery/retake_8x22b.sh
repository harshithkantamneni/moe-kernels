#!/usr/bin/env bash
# Recovery after DRIVER-DONE (8x22B, 2026-09-28): (1) G=8's timed page again at the
# 1710 lock (its first page failed V0 on NATIVE's repeats alone); (2) the deep pages
# as a lock ladder 1710 1605 1500 (1710 slipped at G=4 tread >= 7 on the module's
# software power cap, 0x4, at ~345 W GPU draw). Predictions at a lower lock are the
# 1710 registration scaled by the measured clock law (eta ~1): computed after it.
set -uo pipefail
. ~/moe/env.sh && cd "$REPO" || exit 2
export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
S=$SESSION_ROOT; ST=$S/gh200-driver/status; B=$(cat "$S/base-tag.txt")
R3=(--model mixtral-8x22b --block-m 32 --repeats 9 --duty 0.25 --seed 0)
log() { echo "$(date -u +%FT%TZ) RECOVERY $*" >> "$ST"; }
push() { bash "$REPO/scripts/vm_results_push.sh" push --message "$1" >/dev/null 2>&1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "GPU busy: stop"; exit 2; }
moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null
log "start: G=8 timed at 1710 (tag $B-p2b), then deep G=4 2 on the ladder 1710 1605 1500 (tag $B-deepb)"
python3 scripts/locked_r3.py --session-tag "$B-p2b" --groups 8 --locks 1710 --run-cap-s 3600 -- "${R3[@]}" --treads 6 \
  > "$S/logs/locked_r3-$B-p2b.log" 2>&1
rc=$?; log "G=8 retake locked_r3 exit $rc: $(grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' $S/locked-r3/$B-p2b/status | cut -c22-130)"
sudo -n nvidia-smi -rgc >/dev/null; push "recovery: G=8 retake exit $rc"
python3 scripts/locked_r3.py --session-tag "$B-deepb" --groups 4 2 --locks 1710 1605 1500 --run-cap-s 3600 -- "${R3[@]}" --treads 9 \
  > "$S/logs/locked_r3-$B-deepb.log" 2>&1
rc=$?; log "deep ladder locked_r3 exit $rc"
grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' "$S/locked-r3/$B-deepb/status" | cut -c1-170 | while read -r l; do log "deep: $l"; done
sudo -n nvidia-smi -rgc >/dev/null
log "END"; push "recovery: deep ladder exit $rc; END"
