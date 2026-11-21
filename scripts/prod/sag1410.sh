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
#
# PRECONDITION (review 2, MINOR 5): the bytes pages read the preflight's census, $SESSION_ROOT/census.json (setup_vm's
# preflight PF6, the driver's census_for for the study's default model). No census: FAIL here, before the driver runs
# (the driver would refuse each bytes unit one by one). The par needs the driver's exit 0 and a gate record written
# after this step began (sag_par.py --started; MINOR 4).
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
census_check() {   # census_check SESSION_ROOT: the preflight's census (PF6) is there and non-empty
  [[ -s "$1/census.json" ]] || { echo "FAIL sag1410: no census at $1/census.json (setup_vm's preflight PF6 writes it)"; return 3; }
  echo "PASS sag1410 precondition: census $1/census.json"
}
if [[ $DRY -eq 1 ]]; then
  echo "precondition: \$SESSION_ROOT/census.json (PF6) non-empty, else FAIL before the driver"
  if [[ -n "${SESSION_ROOT:-}" ]]; then census_check "$SESSION_ROOT" || exit 3; fi
  set +e; bash "$DRV" --dry-run --plan "$PLAN"; rc=$?; set -e
  [[ $rc -eq 2 ]] || { echo "FAIL sag1410: the driver's dry run exited $rc"; exit 3; }
  echo "PASS sag1410 plan (the driver's dry run)"
  exit 0
fi
: "${RESULTS_ROOT:?FAIL sag1410: RESULTS_ROOT unset (source ~/moe/env.sh)}"
: "${SESSION_ROOT:?FAIL sag1410: SESSION_ROOT unset (source ~/moe/env.sh)}"
census_check "$SESSION_ROOT" || exit 3
# ONE SAG PER RUN (v10 blind run 2026-11-21): blind.sh chains models on one board, and the driver keys its f128 pages
# by date and model (<date>-<card>-<model>-f128-r3-counters, remembered in $SESSION_ROOT/counters-dir-<model>-f128.txt),
# so a second sag on the board wrote into the first one's page directory and the finaliser's glob read both. Before
# the driver runs, an earlier sag's pages, pointers and gate record are MOVED ASIDE (renamed with .aside-<UTC>, never
# deleted; the earlier run's par.json keeps each page's sha256), so this run measures into fresh directories.
aside="aside-$(date -u +%Y%m%dT%H%M%SZ)"
for f in "$SESSION_ROOT"/counters-dir-*-f128.txt; do
  [[ -f "$f" ]] || continue
  d="$(cat "$f")"
  if [[ -n "$d" && -d "$d" ]]; then mv "$d" "$d.$aside"; echo "sag1410: moved an earlier sag's pages aside: $d.$aside"; fi
  mv "$f" "$f.$aside"
done
for d in "$RESULTS_ROOT"/*-f128-r3-counters; do
  [[ -d "$d" ]] || continue
  mv "$d" "$d.$aside"; echo "sag1410: moved an earlier sag's pages aside: $d.$aside"
done
rec="$SESSION_ROOT/gh200-driver/gate-b3sag-1410.json"
if [[ -e "$rec" ]]; then mv "$rec" "$rec.$aside"; echo "sag1410: moved an earlier sag gate record aside: $rec.$aside"; fi
started=$(date +%s)
deadline=$(( started + 60 * ${SAG_DEADLINE_MIN:-75} ))
rc=0
bash "$DRV" --plan "$PLAN" --deadline "$deadline" --no-push || rc=$?
echo "sag1410: driver exit $rc"
mkdir -p "$OUT"
"${PY_BASE:-${PY_VLLM:-python3}}" "$HERE/sag_par.py" --results "$RESULTS_ROOT" \
  --record "$SESSION_ROOT/gh200-driver/gate-b3sag-1410.json" --driver-exit "$rc" --started "$started" --out "$OUT/par.json"
