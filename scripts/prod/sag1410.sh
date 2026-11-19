#!/usr/bin/env bash
# The production check's on-board 128x128 sag par at 1410 MHz (review 1, BLOCKER 2), by the v7 prefill blind's own
# machinery, unchanged: scripts/gh200_v7p_session.sh runs the four units of scripts/prod/sag1410.plan (the prelude at
# the 1410 lock and 700 W, the OLMoE and Mixtral-8x7B f128 bytes pages, gate b3sag), then scripts/prod/sag_par.py runs
# v7pcommon.finalise_pars on what it left and writes OUT/par.json (score_prod.py --sag).
#
#   . ~/moe/env.sh && bash scripts/prod/sag1410.sh --out "$R/sag"        # main.sh / smoke.sh step sag1410
#   bash scripts/prod/sag1410.sh --out DIR --dry-run                        # the driver's own dry run of the plan
#
# The driver writes its pages under $RESULTS_ROOT and its ledger and gate record under $SESSION_ROOT/gh200-driver
# (env.sh); it runs with --no-push (main.sh pushes after its measurement blocks). Exit: sag_par.py's (0 PASS, 3 FAIL);
# a FAIL is a hard stop in main.sh (the prefill branch has no par without it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "FAIL sag1410: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "FAIL sag1410: --out DIR" >&2; exit 2; }
DRV="$REPO/scripts/gh200_v7p_session.sh"
PLAN="scripts/prod/sag1410.plan"
cd "$REPO"
if [[ $DRY -eq 1 ]]; then
  set +e; bash "$DRV" --dry-run --plan "$PLAN"; rc=$?; set -e
  [[ $rc -eq 2 ]] || { echo "FAIL sag1410: the driver's dry run exited $rc"; exit 3; }
  echo "PASS sag1410 plan (the driver's dry run)"
  exit 0
fi
: "${RESULTS_ROOT:?FAIL sag1410: RESULTS_ROOT unset (source ~/moe/env.sh)}"
: "${SESSION_ROOT:?FAIL sag1410: SESSION_ROOT unset (source ~/moe/env.sh)}"
deadline=$(( $(date +%s) + 60 * ${SAG_DEADLINE_MIN:-75} ))
rc=0
bash "$DRV" --plan "$PLAN" --deadline "$deadline" --no-push || rc=$?
echo "sag1410: driver exit $rc"
mkdir -p "$OUT"
"${PY_BASE:-${PY_VLLM:-python3}}" "$HERE/sag_par.py" --results "$RESULTS_ROOT" \
  --record "$SESSION_ROOT/gh200-driver/gate-b3sag-1410.json" --driver-exit "$rc" --out "$OUT/par.json"
