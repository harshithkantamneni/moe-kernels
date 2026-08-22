#!/usr/bin/env bash
# Commit a curated result set back to the repo, so results leave the pod.
#
#   bash scripts/publish_results.sh                       # newest run
#   bash scripts/publish_results.sh --run-id abc123def    # a specific run
#   bash scripts/publish_results.sh --label first-smoke   # name the directory
#
# results/ is gitignored on purpose: raw runs are large, machine-specific, and
# regenerable. results/published/ is tracked on purpose: it is where a run you
# chose to keep goes, alongside the hardware calibration it was measured
# against. Publishing is therefore a decision, not a side effect.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
RESULTS_DIR="${MOE_RESULTS_DIR:-$WORKSPACE/results}"
[[ -d "$RESULTS_DIR" ]] || RESULTS_DIR="$REPO_ROOT/results"

RUN_ID=""
LABEL=""
PUSH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)  RUN_ID="$2"; shift 2 ;;
    --label)   LABEL="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[publish] %s\n' "$*"; }

# Newest CSV wins if no run id was given.
if [[ -z "$RUN_ID" ]]; then
  newest="$(ls -t "$RESULTS_DIR"/run_*.csv 2>/dev/null | head -1 || true)"
  [[ -n "$newest" ]] || { echo "[publish] no run_*.csv in $RESULTS_DIR" >&2; exit 1; }
  RUN_ID="$(basename "$newest" | sed -E 's/^run_([^_]+)_.*/\1/')"
  log "no --run-id given; using the newest run: $RUN_ID"
fi

files=("$RESULTS_DIR"/run_"$RUN_ID"_*.csv)
[[ -e "${files[0]}" ]] || { echo "[publish] nothing matches run_${RUN_ID}_*.csv" >&2; exit 1; }

STAMP="$(date -u +%Y-%m-%d)"
DEST="results/published/${STAMP}-${LABEL:-run-$RUN_ID}"
mkdir -p "$DEST"

log "copying $(ls "$RESULTS_DIR"/run_"$RUN_ID"_*.csv | wc -l) CSV(s) and manifest(s)"
cp "$RESULTS_DIR"/run_"$RUN_ID"_*.csv "$DEST"/
cp "$RESULTS_DIR"/run_"$RUN_ID"_*.manifest.jsonl "$DEST"/ 2>/dev/null || true
[[ -f "$RESULTS_DIR/merged.csv" ]] && cp "$RESULTS_DIR/merged.csv" "$DEST"/ || true

# The calibration is not optional context: every efficiency column in these rows
# is quoted against it, so a result set without it cannot be interpreted later.
if [[ -f moe/bench/hardware/measured.yaml ]]; then
  cp moe/bench/hardware/measured.yaml "$DEST"/measured.yaml
  log "included the hardware calibration these rows were measured against"
else
  log "WARNING: no measured.yaml; efficiency columns will be uninterpretable"
fi

# A summary a human can read without opening the CSV.
PY="${MOE_PYTHON:-$WORKSPACE/venvs/base/bin/python}"
[[ -x "$PY" ]] || PY="python3"
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
    print(f"- gpu: {sorted({r.get('gpu_name','') for r in rows}) }")
    print(f"- commit: {sorted({r.get('git_sha','')[:12] for r in rows})}")
    dirty = {r.get("git_dirty") for r in rows}
    if "True" in dirty:
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
print("| impl | model | tokens | ms p50 | TFLOP/s | AI | tile eff @128 |")
print("|---|---|---:|---:|---:|---:|---:|")
for key in sorted(best)[:60]:
    r = min(best[key], key=lambda x: row_float(x, "ms_p50"))
    print(f"| {key[0]} | {key[1]} | {key[2]} | {row_float(r,'ms_p50'):.4f} | "
          f"{row_float(r,'tflops'):.1f} | {row_float(r,'arith_intensity_compulsory'):.1f} | "
          f"{row_float(r,'load_tile_eff_bm128'):.3f} |")
SUMMARY

log "wrote $DEST/SUMMARY.md"
git add "$DEST"
if git diff --cached --quiet; then
  log "nothing new to commit"
  exit 0
fi
git -c user.email="${GIT_AUTHOR_EMAIL:-hkantamneni2@wisc.edu}" \
    -c user.name="${GIT_AUTHOR_NAME:-Harshith Kantamneni}" \
    commit -q -m "Results: ${LABEL:-run $RUN_ID} on $(hostname)"
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
