#!/usr/bin/env bash
# Recovery after the driver, decided from its own ledger: every timed or deep page that came back
# INVALID or slipped its lock is taken once more (1710 lock; a slipped deep page on the ladder
# 1710 1605 1500). Model and copies from $1 (the driver's --model); tags end -b.
set -uo pipefail
M="$1"
. ~/moe/env.sh && cd "$REPO" || exit 2
export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
S=$SESSION_ROOT; ST=$S/gh200-driver/status; B=$(cat "$S/base-tag.txt")
R3=(--model "$M" --block-m 32 --repeats 9 --duty 0.25 --seed 0 --declared-copies 9)
log() { echo "$(date -u +%FT%TZ) RECOVERY $*" >> "$ST"; }
push() { bash "$REPO/scripts/vm_results_push.sh" push --message "$1" >/dev/null 2>&1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "GPU busy: stop"; exit 2; }
bad() { grep -E "^[^ ]+ $1: .* G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit [0-9]+ (INVALID|on SIGKILL)|^[^ ]+ $1: .* G=[0-9]+ slipped" "$ST" | sed -E 's/.* G=([0-9]+) .*/\1/' | sort -un | tr '\n' ' '; }
TIMED=$(bad timed); DEEP=$(bad deep)
log "start: timed G to retake [${TIMED}], deep G to retake [${DEEP}]"
for G in $TIMED; do
  t=6; [ "$G" = 3 ] && t=8
  moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null
  python3 scripts/locked_r3.py --session-tag "$B-t${G}b" --groups "$G" --locks 1710 --run-cap-s 3600 -- "${R3[@]}" --treads $t \
    > "$S/logs/locked_r3-$B-t${G}b.log" 2>&1
  log "timed G=$G retake exit $?: $(grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' $S/locked-r3/$B-t${G}b/status | cut -c22-150)"
  sudo -n nvidia-smi -rgc >/dev/null; push "recovery: timed G=$G retake"
done
if [ -n "$DEEP" ]; then
  locks="1710"; grep -qE "deep: .* slipped" "$ST" && locks="1710 1605 1500"
  moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null
  python3 scripts/locked_r3.py --session-tag "$B-deepb" --groups $DEEP --locks $locks --run-cap-s 3600 -- "${R3[@]}" --treads 9 \
    > "$S/logs/locked_r3-$B-deepb.log" 2>&1
  log "deep retake (G $DEEP, locks $locks) exit $?"
  grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' "$S/locked-r3/$B-deepb/status" | cut -c1-170 | while read -r l; do log "deep: $l"; done
  sudo -n nvidia-smi -rgc >/dev/null
fi
log "END"; push "recovery: END"
