#!/usr/bin/env bash
# Commit a curated result set back to the repo, so results leave the pod.
#
#   bash scripts/publish_results.sh                          # the only run present
#   bash scripts/publish_results.sh --all                    # every run in the dir
#   bash scripts/publish_results.sh --run-id a --run-id b    # exactly these
#   bash scripts/publish_results.sh --label first-smoke      # name the directory
#   bash scripts/publish_results.sh --dry-run                # stage it, touch no git
#
# results/ is gitignored on purpose: raw runs are large, machine-specific, and
# regenerable. results/published/ is tracked on purpose: it is where a run you
# chose to keep goes, alongside the hardware calibration it was measured
# against. Publishing is therefore a decision, not a side effect.
#
# A SWEEP IS NOT A RUN. Each venv in a sweep writes its own run_<id>_<env>.csv,
# so a three-framework sweep leaves three run ids behind and all three are one
# experiment. This used to publish the newest CSV, which published a third of
# the result and looked complete; the base and vLLM arms of the 2026-08-26
# three-way had to be committed by hand afterwards. It now refuses to guess.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
RESULTS_DIR="${MOE_RESULTS_DIR:-$WORKSPACE/results}"
[[ -d "$RESULTS_DIR" ]] || RESULTS_DIR="$REPO_ROOT/results"
PUBLISH_ROOT="${MOE_PUBLISH_ROOT:-$REPO_ROOT/results/published}"

# One interpreter for the whole script. The device probe used to shell out to a
# bare `python3`, which is a different environment from the one that reads the
# rows, and on a pod it is usually the one without the dependencies.
PY="${MOE_PYTHON:-$WORKSPACE/venvs/base/bin/python}"
[[ -x "$PY" ]] || PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

RUN_IDS=()
ALL=0
LABEL=""
PUSH=1
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)  RUN_IDS+=("$2"); shift 2 ;;
    --all)     ALL=1; shift ;;
    --label)   LABEL="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[publish] %s\n' "$*"; }
die() { printf '[publish] %s\n' "$*" >&2; exit 1; }

# Every run id with a CSV in the directory, in the order the files were written.
# A read loop rather than mapfile: macOS ships bash 3.2, where mapfile does not
# exist, and this script is read and run on the laptop as often as on the pod.
PRESENT=()
while IFS= read -r id; do
  [[ -n "$id" ]] && PRESENT+=("$id")
done < <(
  ls -tr "$RESULTS_DIR"/run_*.csv 2>/dev/null \
    | sed -E 's|.*/run_([^_]+)_.*|\1|' | awk '!seen[$0]++'
)
(( ${#PRESENT[@]} )) || die "no run_*.csv in $RESULTS_DIR"

if (( ALL )); then
  RUN_IDS=("${PRESENT[@]}")
elif (( ${#RUN_IDS[@]} == 0 )); then
  if (( ${#PRESENT[@]} == 1 )); then
    RUN_IDS=("${PRESENT[0]}")
    log "one run present; using $RUN_IDS"
  else
    # Taking the newest here is what published a third of a sweep. results/ also
    # outlives a session, so the several runs present may be one sweep or may be
    # unrelated experiments, and this script cannot tell which from the files.
    {
      echo "[publish] $RESULTS_DIR holds ${#PRESENT[@]} runs, and which of them belong"
      echo "[publish] together is a question about your session, not about the files:"
      for id in "${PRESENT[@]}"; do
        printf '[publish]     %s  (%s)\n' "$id" \
          "$(ls "$RESULTS_DIR"/run_"$id"_*.csv | sed -E 's|.*_([^_]+)\.csv|\1|' | paste -sd, -)"
      done
      echo "[publish] publish the whole directory:  --all"
      echo "[publish] or name the arms of one sweep: --run-id ${PRESENT[0]} --run-id ${PRESENT[1]}"
    } >&2
    exit 1
  fi
fi

FILES=()
for id in "${RUN_IDS[@]}"; do
  matches=("$RESULTS_DIR"/run_"$id"_*.csv)
  [[ -e "${matches[0]}" ]] || die "nothing matches run_${id}_*.csv in $RESULTS_DIR"
  FILES+=("${matches[@]}")
done
log "publishing ${#RUN_IDS[@]} run(s), ${#FILES[@]} CSV(s): ${RUN_IDS[*]}"

# The device is read from the ROWS, not from this machine: a result set belongs
# to the GPU that produced it, and the same harness now runs on several. Each
# device therefore gets its own published arm, with its own calibration beside
# it. measured_slug is imported rather than reimplemented so the directory name
# and the calibration filename cannot drift apart.
read -r GPU SLUG < <("$PY" - "${FILES[@]}" <<'DEVICE'
import csv, sys
from moe.bench.roofline import measured_slug

names = set()
for path in sys.argv[1:]:
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("gpu_name"):
                names.add(row["gpu_name"])
if len(names) > 1:
    # One arm carries one calibration, and every efficiency column in these rows
    # is quoted against it. Two devices in one directory have no single answer.
    sys.exit("[publish] these runs span more than one GPU, so they cannot share "
             "one published arm and one calibration:\n[publish]     "
             + "\n[publish]     ".join(sorted(names))
             + "\n[publish] publish each device separately with --run-id.")
gpu = names.pop() if names else "unknown-device"
print(gpu.replace(" ", "\x1f"), measured_slug(gpu))
DEVICE
) || exit 1
GPU="${GPU//$'\x1f'/ }"
log "rows were measured on $GPU"

STAMP="$(date -u +%Y-%m-%d)"
SUFFIX="${LABEL:-run-${RUN_IDS[0]}}"
DEST="$PUBLISH_ROOT/${STAMP}-${SLUG#measured_}-${SUFFIX}"
mkdir -p "$DEST"

cp "${FILES[@]}" "$DEST"/
for id in "${RUN_IDS[@]}"; do
  cp "$RESULTS_DIR"/run_"$id"_*.manifest.jsonl "$DEST"/ 2>/dev/null || true
done

# merged.csv is REBUILT from the selected arms rather than copied. results/
# persists across sessions, so the merged file in it can hold run ids from
# experiments that have nothing to do with this one; copying it wholesale is how
# 716 rows measured against a different calibration ended up inside a published
# arm, where nothing downstream could tell them apart.
"$PY" - "$DEST" <<'MERGE'
import pathlib, sys
from moe.bench.schema import merge_csvs

dest = pathlib.Path(sys.argv[1])
arms = sorted(dest.glob("run_*.csv"))
n = merge_csvs(arms, dest / "merged.csv")
print(f"[publish] rebuilt merged.csv from {len(arms)} arm(s): {n} rows")
MERGE

# The calibration is not optional context: every efficiency column in these rows
# is quoted against it, so a result set without it cannot be interpreted later.
CAL="moe/bench/hardware/${SLUG}.yaml"
if [[ -f "$CAL" ]]; then
  cp "$CAL" "$DEST"/measured.yaml
  log "included $CAL, the calibration these rows were measured against"
else
  log "WARNING: no $CAL; efficiency columns will be uninterpretable"
fi

# Figures belong with the data they were drawn from, not in the repo root.
if [[ -d plots ]]; then
  mkdir -p "$DEST/plots"
  cp plots/*.png "$DEST/plots/" 2>/dev/null && \
    log "included $(ls "$DEST/plots" | wc -l | tr -d ' ') figure(s)" || true
fi

# A summary a human can read without opening the CSV.
"$PY" - "$DEST" > "$DEST/SUMMARY.md" <<'SUMMARY'
import sys, collections, pathlib
sys.path.insert(0, ".")
from moe.bench.schema import passed, read_csv, row_float

dest = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(dest.glob("run_*.csv")):
    rows.extend(read_csv(p))

print(f"# Results: {dest.name}\n")
print(f"- rows: {len(rows)}")
ok = [r for r in rows if passed(r)]
print(f"- correctness passed: {len(ok)} / {len(rows)}")
if rows:
    # Per arm, so a sweep that lost one venv is visible here rather than only in
    # a row count that looks plausible on its own.
    per_run = collections.Counter(r.get("run_id", "") for r in rows)
    print(f"- arms: {len(per_run)}")
    for rid, n in sorted(per_run.items()):
        envs = sorted({r.get("env_name", "") for r in rows if r.get("run_id") == rid})
        print(f"    - `{rid}` {'+'.join(envs)}: {n} rows")
    print(f"- implementations: {sorted({r.get('impl','') for r in rows})}")
    print(f"- gpu: {sorted({r.get('gpu_name','') for r in rows})}")
    print(f"- commit: {sorted({r.get('git_sha','')[:12] for r in rows})}")
    if "True" in {r.get("git_dirty") for r in rows}:
        print("- **WARNING: some rows were measured from a dirty working tree**")
throttled = [r for r in rows if r.get("throttled") == "True"]
if throttled:
    print(f"- **{len(throttled)} rows throttled (clocks dropped >5% mid-cell)**")

fails = [r for r in rows if not passed(r)]
if fails:
    print("\n## Correctness failures\n")
    for r in fails[:20]:
        print(f"- `{r['impl']}` {r['model']}/T{r['num_tokens']} "
              f"rel={row_float(r,'rel_err'):.3e} tol={row_float(r,'tol_rel_max'):.3e}")

print("\n## Fastest per (impl, model, tokens), L2-flushed eager rows\n")
best = collections.defaultdict(list)
for r in ok:
    if r.get("l2_flush") == "True" and r.get("cuda_graph") == "False":
        best[(r["impl"], r["model"], int(row_float(r, "num_tokens")))].append(r)
print("| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |")
print("|---|---|---|---:|---:|---:|---:|")
for key in sorted(best)[:60]:
    r = min(best[key], key=lambda x: row_float(x, "ms_p50"))
    # `covers` is in the table because these ms are NOT comparable across rows
    # with different extents: one GEMM against a five-stage fused block.
    print(f"| {key[0]} | {r.get('covers','')} | {key[1]} | {key[2]} | "
          f"{row_float(r,'ms_p50'):.4f} | {row_float(r,'tflops'):.1f} | "
          f"{row_float(r,'arith_intensity_compulsory'):.1f} |")
SUMMARY

log "wrote $DEST/SUMMARY.md"

if (( DRY )); then
  log "dry run: $DEST is staged on disk, nothing was committed"
  exit 0
fi

git add "$DEST"
if git diff --cached --quiet; then
  log "nothing new to commit"
  exit 0
fi
git -c user.email="${GIT_AUTHOR_EMAIL:-hkantamneni2@wisc.edu}" \
    -c user.name="${GIT_AUTHOR_NAME:-Harshith Kantamneni}" \
    commit -q -m "Results: ${LABEL:-${#RUN_IDS[@]} run(s)} on $(hostname)"
log "committed $DEST"

if (( PUSH )); then
  if git push -q origin HEAD 2>/dev/null; then
    log "pushed. The result set is now on GitHub."
  else
    log "push FAILED (no credentials on this pod?)"
    log "  run: gh auth login    then: git push origin HEAD"
    log "  the commit is safe locally either way"
  fi
fi
