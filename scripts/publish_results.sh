#!/usr/bin/env bash
# Commit a curated result set back to the repo, so results leave the pod.
#
#   bash scripts/publish_results.sh                          # the only run present
#   bash scripts/publish_results.sh --all                    # every run in the dir
#   bash scripts/publish_results.sh --run-id a --run-id b    # exactly these
#   bash scripts/publish_results.sh --label first-smoke      # name the directory
#   bash scripts/publish_results.sh --dry-run                # stage it, touch no git
#   bash scripts/publish_results.sh --allow-foreign-calibration   # see below
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
#
# THE CALIBRATION IS COPIED FROM ONE FILE PER DEVICE, AND IT MOVES UNDER YOU.
# `moe/bench/hardware/measured_<device>.yaml` is overwritten by every new
# calibration, and the copy below takes whatever is in it right now -- not what
# the rows were measured against. On 2026-08-28 the whole-layer sweep ran
# 18:08-19:21 UTC, a recalibration overwrote that file at 19:29, and this script
# published the arm at 19:35 with the new ruler beside rows stamped from the old
# one. The two disagree by 9.9% on the compute ceiling, nothing downstream could
# tell, and claim C5 lost its target for three days.
#
# So the copy is now CHECKED against the ceilings the rows themselves carry, and
# a mismatch stops the publish. `--allow-foreign-calibration` is for the one case
# where a foreign calibration is the point: `scripts/recompute_ceilings.py`
# derives an arm by restamping old rows against a NEW calibration, and its output
# declares itself derived so it needs no flag at all. The flag exists for a
# deliberate case that has not declared itself yet, and it writes the admission
# into the published arm rather than letting it pass in silence.
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
REPORTS=()
ALL=0
LABEL=""
PUSH=1
DRY=0
FOREIGN_CAL=0
MISSING_SHA_REASON=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)  RUN_IDS+=("$2"); shift 2 ;;
    --reports) REPORTS+=("$2"); shift 2 ;;
    --all)     ALL=1; shift ;;
    --label)   LABEL="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --dry-run) DRY=1; shift ;;
    --allow-foreign-calibration) FOREIGN_CAL=1; shift ;;
    --allow-missing-sha) MISSING_SHA_REASON="${2:?--allow-missing-sha needs a reason, which is written into SUMMARY.md}"; shift 2 ;;
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
  # Deliberately NOT "the calibration these rows were measured against". That is
  # what this copy is supposed to be and what it silently was not for the
  # whole-layer arm; the check below is what turns it into a claim worth making.
  log "included $CAL, the calibration currently on disk for this device"
else
  log "WARNING: no $CAL; efficiency columns will be uninterpretable"
fi

# Does the file just copied actually belong to this sweep? The rows carry the
# ceilings they were computed against, so they can answer that themselves.
if ! "$PY" - "$DEST" "$FOREIGN_CAL" <<'PROVENANCE'
import pathlib, sys
from moe.bench.published import calibration_provenance, entitled_ridge

dest, override = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
prov = calibration_provenance(dest)
ridge, why = entitled_ridge(dest)

print(f"[publish] calibration provenance: {prov.verdict}")
for key in ("ceilings", "commit", "date", "device"):
    if key in prov.evidence:
        print(f"[publish]   {key:9} {prov.evidence[key]}")
print(f"[publish]   ridge     {why}")

# The verdict goes INTO the arm, pass or fail, so the next reader inherits it
# instead of re-deriving it from md5sums three days later.
lines = [f"# Calibration provenance: {dest.name}", "",
         f"Verdict: **{prov.verdict}**", ""]
if prov.derived_from:
    lines += [f"Declared derived from `{prov.derived_from}`, so a calibration "
              f"from a later session is expected here.", ""]
for key, value in prov.evidence.items():
    lines.append(f"- `{key}`: {value}")
lines += ["", f"Ridge: {why}", ""]
if prov.blocking_reason:
    lines += [f"BLOCKING: {prov.blocking_reason}", ""]
    if override:
        lines += ["Published anyway with `--allow-foreign-calibration`. Every "
                  "efficiency column in these rows is quoted against a ruler "
                  "that is not the one they were measured with.", ""]
(dest / "CALIBRATION_PROVENANCE.md").write_text("\n".join(lines))

if prov.blocking_reason and not override:
    sys.exit(f"[publish] REFUSING TO PUBLISH: {prov.blocking_reason}\n"
             "[publish] The rows and the calibration beside them disagree, so\n"
             "[publish] every efficiency column in this arm would be quoted\n"
             "[publish] against a ruler it was not measured with. Either copy in\n"
             "[publish] the calibration these rows actually used, or re-derive\n"
             "[publish] the arm with scripts/recompute_ceilings.py, which\n"
             "[publish] restamps the rows and declares itself. Pass\n"
             "[publish] --allow-foreign-calibration only if the mismatch is the\n"
             "[publish] point and you want it recorded in the arm.")
if prov.blocking_reason:
    print("[publish] OVERRIDDEN with --allow-foreign-calibration; the admission "
          "is in CALIBRATION_PROVENANCE.md")
PROVENANCE
then
  log "$DEST is staged on disk and was NOT committed"
  exit 1
fi

# --------------------------------------------------------------------------
# the report/cells pairs this arm carries
# --------------------------------------------------------------------------
# `find results/published -name cells.csv` returned ZERO across every published
# arm. `scaled_iters` promises that the CSV shows which numbers rest on 5
# samples and which on 50, and the file it promises never reached the repo, so
# the only per-cell p50s that survive anywhere are in one session log. A report
# without its cells is a verdict without the measurement under it.
#
# The experiment scripts write `<dir>/report.json` beside `<dir>/cells.csv`, so
# they travel together or not at all: a report copied without its cells is
# refused here rather than discovered missing by a reader three days later.
for rdir in "${REPORTS[@]+"${REPORTS[@]}"}"; do
  stem="$(basename "${rdir%/}")"
  [[ -f "$rdir/report.json" ]] || die "no report.json in $rdir"
  [[ -f "$rdir/cells.csv" ]] || die \
    "$rdir has a report.json and no cells.csv. Publishing the verdict without
[publish] the measurements under it is how every published report came to have
[publish] no cells beside it. Re-run the arm, or publish the directory that has
[publish] both."
  cp "$rdir/report.json" "$DEST/${stem}.report.json"
  cp "$rdir/cells.csv" "$DEST/${stem}.cells.csv"
  log "included report + cells for $stem"
done

# Figures belong with the data they were drawn from, AND THEY MUST BE DRAWN FROM
# IT. This used to copy every PNG out of the repo-root `plots/` directory that
# `run_all.sh` had filled from the whole `/workspace/results` volume: the
# ridge-resolution arm shipped a `scaling_toy_bf16.png` its CSVs do not contain,
# and the bf16-only alpha-0558 arm shipped six fp8 figures byte-identical to
# another arm's. Commit bd8b5b4 said the figures were "regenerable from the
# committed CSVs"; they were not, and no reader could have told. plot.py is now
# pointed at the arm, so a figure that is not in these rows cannot be drawn.
if "$PY" scripts/plot.py --results "$DEST" --out "$DEST/plots" >/dev/null 2>&1; then
  log "drew $(ls "$DEST/plots" 2>/dev/null | wc -l | tr -d ' ') figure(s) from this arm's own CSVs"
else
  log "plotting skipped (matplotlib absent, or these rows support no figure)"
  rmdir "$DEST/plots" 2>/dev/null || true
fi

# --------------------------------------------------------------------------
# every row's commit must exist
# --------------------------------------------------------------------------
# A row's git_sha is the whole of its attribution and nothing checked it.
# 2,100 published rows cite a commit that exists nowhere, rewritten by the very
# `git pull --rebase` this script recommends when a push is rejected. The check
# is scoped to THIS arm, so the historical miss does not block a new publish,
# and the report goes into the arm either way.
#
# WHY THIS REFUSES THE COMMIT AND NOT THE STAGING. A dry run's contract is
# "stage it, touch no git", and the finding is worth more staged than withheld:
# the report goes into the arm and into SUMMARY.md, where the operator reads it
# before deciding. What must not happen quietly is the COMMIT, so that is what
# is refused. A calibration mismatch is refused earlier and harder because it
# makes the arm's numbers uninterpretable; an unresolvable sha leaves the
# numbers readable and their provenance uncheckable, and the remedy (push the
# commit that made them) belongs to the act of publishing.
SHA_REPORT="$DEST/GIT_SHA_CHECK.txt"
SHA_OK=1
if "$PY" scripts/check_published_shas.py --arm "$DEST" --repo "$REPO_ROOT" \
     > "$SHA_REPORT" 2>&1; then
  log "every recorded git_sha in this arm resolves"
else
  SHA_OK=0
  sed 's/^/[publish]   /' "$SHA_REPORT"
fi

# AND THE SAME QUESTION FROM A STRANGER'S SIDE, ASKED BEFORE THE COMMIT.
# A sha that resolves HERE and on no remote branch is one a stranger who clones
# cannot reach, which is the same nothing as a missing commit from their side.
#
# THIS USED TO RUN AFTER THE PUSH AND APPEND TO A FILE ALREADY COMMITTED. Two
# things were wrong with that. Every successful publish ended with a TRACKED
# file modified in the working tree, which is the dirty-tree defect this whole
# session shell is being repaired for: the next thing measured on that pod
# stamps git_dirty=True on every row. And the verdict reached neither the
# committed arm nor SUMMARY.md, which quotes this file as it stands when the
# summary is generated, so the one reader it was written for never saw it.
#
# Asked before the push, the honest answer for a sha this very publish is about
# to push would be UNPUSHED, which is why --will-push exists: the checker
# reports PENDING for a sha that `git push origin HEAD` will place on the
# remote, and UNPUSHED only for one that nothing here will. If the push then
# fails, the failure branch below says so rather than leaving PENDING to be read
# as done. Advisory either way: the rows are readable, only their reachability
# is in question.
REMOTE_ARGS=(--require-remote)
if (( PUSH && ! DRY )); then
  REMOTE_ARGS+=(--will-push HEAD)
fi
{ echo; echo "# reachability from a remote, asked before this publish committed"; } \
  >> "$SHA_REPORT"
if "$PY" scripts/check_published_shas.py --arm "$DEST" --repo "$REPO_ROOT" \
     "${REMOTE_ARGS[@]}" >> "$SHA_REPORT" 2>&1; then
  log "every git_sha in this arm is on a remote, or on the HEAD this publish pushes"
else
  # Deliberately not asserting which gate failed: this invocation re-scores
  # resolvability as well, so its exit code is also 1 for the MISSING case the
  # block above already reported, and a line that named the remote as the cause
  # would be wrong half the time.
  log "WARNING: the reachability check did not come back clean. Either a sha is"
  log "  on no remote branch and nothing here will push it, or it does not"
  log "  resolve at all. Either way a stranger who clones cannot reach the code"
  log "  those rows name. Verdicts in $SHA_REPORT, and in SUMMARY.md, which"
  log "  quotes it. Reported and not blocking: the numbers are readable."
fi

# A summary a human can read without opening the CSV.
"$PY" - "$DEST" "$MISSING_SHA_REASON" > "$DEST/SUMMARY.md" <<'SUMMARY'
import sys, collections, pathlib, statistics
sys.path.insert(0, ".")
from moe.bench.schema import passed, read_csv, row_float

dest = pathlib.Path(sys.argv[1])
missing_sha_reason = sys.argv[2] if len(sys.argv) > 2 else ""
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


def emit(line):
    print(f"- **{line}**")


# `throttled` IS ONE COLUMN WITH TWO MEANINGS, split by the instrument that
# wrote the row, and until 2026-09-03 this line described neither: "clocks
# dropped >5% mid-cell" named a detector that never existed. On a row written
# before schema v5 the flag is the retired two-sample drift check: the SM clock
# read at an idle instant before the cell and again after it, set on a >5%
# drop. That detected whether the FIRST read had caught the idle boost clock,
# not throttling under load (moe/bench/timing.py, CLOCKS ARE READ UNDER LOAD:
# on the alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged
# and unflagged replicates timed at ratio 0.998). On a v5 row the driver sets
# it when the DRIFT verdict taken WHILE the trials ran failed, or when the
# LEVEL verdict failed on the LOW side (moe/bench/driver.py). LEVEL IS A BAND,
# not a floor: since 03df2d4 (2026-09-03) it is [LEVEL_FRACTION,
# LEVEL_HIGH_FRACTION] around the clock the calibration GEMM ran at, and the
# side of a failure is on the row as `clock_level_side`. A HIGH failure is a
# cell boosted ABOVE the band, the normal state of a memory-bound decode cell
# on an H200 (1980 MHz under memory load against a 1515 MHz bf16 reference);
# it is not throttling, the driver does not write it into `throttled`, and its
# fixed-roof fraction is what is wrong, with pct_of_roof_at_cell_clock as the
# correction. So this line reads the SIDE: it names the band rather than
# "below 95%", reports the side on every flagged LEVEL failure, and reports the
# HIGH rows that were KEPT beside them rather than filing them under the flag.
# Until 2026-09-08 this comment described LEVEL as one-sided ("LEVEL or DRIFT
# ... failed") and the printed line said "below 95% of". The two flag origins
# are counted apart and named by what each detected; a row whose instrument
# cannot be read is reported as exactly that rather than filed under either.
from collections import Counter

from moe.bench.schema import (UNRECORDED, VERDICT_FAILED, TimingInstrumentUnrecorded,
                              has_kernel_timing, row_bool, timing_verdict)
try:
    from moe.bench.timing import DRIFT_FRACTION, LEVEL_FRACTION, LEVEL_HIGH_FRACTION
    level_word = (f"outside the band {LEVEL_FRACTION:.0%} to "
                  f"{LEVEL_HIGH_FRACTION:.0%} of")
    drift_word = f"more than {DRIFT_FRACTION:.0%} apart"
except Exception:  # torch absent: name the constants rather than guess their values
    level_word = "outside the band timing.LEVEL_FRACTION to timing.LEVEL_HIGH_FRACTION of"
    drift_word = "more than timing.DRIFT_FRACTION apart"


def side_of(r):
    # The row's own word, never a constant copied here: the word is written by
    # moe/bench/driver.py from timing.LEVEL_LOW / LEVEL_HIGH and read back as
    # is, so a side this file has never heard of is printed, not misfiled.
    # A v5 row (00f3324 to 03df2d4) carries no side because that instrument
    # had no high edge, so its LEVEL failure is low by the instrument's own
    # definition, not by this file's default; a v6 row with no side is a row
    # the driver did not write, and is printed as unrecorded.
    side = str(r.get("clock_level_side") or "").strip()
    if side and side != UNRECORDED:
        return side
    try:
        version = int(float(r.get("schema_version") or 0))
    except ValueError:
        version = 0
    return "low" if 0 < version < 6 else "unrecorded"


def sides_text(counter):
    return ", ".join(f"{side} {n}" for side, n in sorted(counter.items()))


flagged = [r for r in rows if row_bool(r, "throttled")]
kept_high = Counter()
for r in rows:
    if row_bool(r, "throttled"):
        continue
    try:
        if has_kernel_timing(r) and timing_verdict(r, "clock_level_ok") == VERDICT_FAILED:
            kept_high[side_of(r)] += 1
    except TimingInstrumentUnrecorded:
        continue
if flagged:
    under_load, legacy, unreadable = [], [], []
    level = drift = 0
    level_sides = Counter()
    for r in flagged:
        try:
            if not has_kernel_timing(r):
                legacy.append(r)
                continue
            lv = timing_verdict(r, "clock_level_ok")
            dr = timing_verdict(r, "clock_drift_ok")
        except TimingInstrumentUnrecorded:
            unreadable.append(r)
            continue
        under_load.append(r)
        level += lv == VERDICT_FAILED
        drift += dr == VERDICT_FAILED
        if lv == VERDICT_FAILED:
            level_sides[side_of(r)] += 1
    if under_load:
        emit(f"{len(under_load)} rows carry throttled=True from the under-load clock "
             f"check: LEVEL failed on {level} (SM clock under load {level_word} the "
             f"clock the calibration GEMM ran at; side: {sides_text(level_sides) or 'none'}), "
             f"DRIFT failed on {drift} (first and last under-load samples {drift_word}, "
             f"either direction)")
        odd = {s: n for s, n in level_sides.items() if s != "low"}
        if odd:
            emit(f"{sum(odd.values())} of those rows fail LEVEL on a side other than low "
                 f"({sides_text(odd)}) and still carry throttled=True, which "
                 f"moe/bench/driver.py never writes: a HIGH failure is a boosted cell, "
                 f"not a throttle. This file was not written by that driver or its side "
                 f"column is wrong; do not read its throttled flag as a throttle")
    if legacy:
        emit(f"{len(legacy)} rows carry throttled=True from the retired pre-v5 drift "
             f"flag: two idle-instant SM-clock reads either side of the cell, >5% "
             f"apart. It detected an idle-boost catch, not throttling under load "
             f"(moe/bench/timing.py)")
    if unreadable:
        emit(f"{len(unreadable)} rows carry throttled=True with an unreadable "
             f"instrument column, so which detector set it cannot be said")
if kept_high:
    high = kept_high.get("high", 0)
    if high:
        emit(f"{high} rows failed LEVEL on the HIGH side (SM clock under load boosted "
             f"above the band) and are NOT throttled: KEPT, as the driver keeps them. "
             f"Their fixed-roof fraction (pct_of_achieved_tflops) is not comparable; "
             f"pct_of_roof_at_cell_clock is the column to read for them")
    other = {s: n for s, n in kept_high.items() if s != "high"}
    if other:
        emit(f"{sum(other.values())} rows failed LEVEL on a side other than high "
             f"({sides_text(other)}) without throttled=True, which moe/bench/driver.py "
             f"never writes: a LOW failure is a throttle and is flagged. This file was "
             f"not written by that driver or its side column is wrong")

fails = [r for r in rows if not passed(r)]
if fails:
    print("\n## Correctness failures\n")
    for r in fails[:20]:
        print(f"- `{r['impl']}` {r['model']}/T{r['num_tokens']} "
              f"rel={row_float(r,'rel_err'):.3e} tol={row_float(r,'tol_rel_max'):.3e}")

# WHICH COMMITS THESE ROWS NAME, and whether they exist. Written here because
# SUMMARY.md is the file a reader opens, and "clean at 7eecff4" was
# unverifiable for three days in a file that said nothing about it.
check = dest / "GIT_SHA_CHECK.txt"
if check.is_file():
    print("\n## Commit resolvability\n")
    print("```")
    print(check.read_text().strip())
    print("```")
    if missing_sha_reason:
        print(f"\n**Published with `--allow-missing-sha`.** Reason given: "
              f"{missing_sha_reason}\n")
        print("Some rows above name a commit this repository does not contain, "
              "so the code that produced them cannot be inspected.")

# THE BASIS IS READ OFF THE ROWS, NOT ASSERTED IN THE HEADING. This table used
# to be headed "Fastest per (impl, model, tokens), L2-flushed eager rows" and
# filtered to l2_flush=True, cuda_graph=False. The alpha-0558 arm has ZERO
# flushed rows -- all 3,696 are l2_flush=False -- so its table was EMPTY under a
# heading naming a basis the arm never ran, and any arm that did have flushed
# rows got a best-of table on top of that.
#
# Two changes. The heading names the bases actually present, so an arm that ran
# one basis says so and an arm that ran two is not silently collapsed into one.
# And the number is the MEDIAN with its spread rather than the minimum: picking
# the fastest of N is the best-of report van der Kouwe names, it moves with N,
# and the spread is the only part of it that tells a reader whether the
# difference beside it means anything.
BASES = {("True", "False"): "L2-flushed eager",
         ("False", "False"): "warm-L2 eager",
         ("True", "True"): "L2-flushed cuda-graph",
         ("False", "True"): "warm-L2 cuda-graph"}


def basis_of(r):
    key = (r.get("l2_flush", ""), r.get("cuda_graph", ""))
    return BASES.get(key, f"l2_flush={key[0] or '?'} cuda_graph={key[1] or '?'}")


groups = collections.defaultdict(list)
for r in ok:
    groups[(basis_of(r), r["impl"], r["model"],
            int(row_float(r, "num_tokens")))].append(r)
present = sorted({k[0] for k in groups})
print("\n## Median ms_p50 per (impl, model, tokens), by timing basis\n")
print(f"Bases present in these rows: {', '.join(present) or 'none'}. "
      "The bases are NOT comparable with each other: a flushed row pays a cold "
      "L2 on every call and a graph row pays no launch.\n")
print("| basis | impl | covers | model | tokens | n | ms p50 median | spread % | TFLOP/s | AI |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
for key in sorted(groups)[:80]:
    cell = groups[key]
    ms = sorted(row_float(r, "ms_p50") for r in cell)
    med = statistics.median(ms)
    spread = 100.0 * (ms[-1] - ms[0]) / med if med else 0.0
    r = cell[len(cell) // 2]
    # `covers` is in the table because these ms are NOT comparable across rows
    # with different extents: one GEMM against a five-stage fused block.
    print(f"| {key[0]} | {key[1]} | {r.get('covers','')} | {key[2]} | {key[3]} | "
          f"{len(cell)} | {med:.4f} | {spread:.1f} | "
          f"{statistics.median([row_float(x,'tflops') for x in cell]):.1f} | "
          f"{row_float(r,'arith_intensity_compulsory'):.1f} |")
SUMMARY

log "wrote $DEST/SUMMARY.md"

# THE INVARIANT, checked on the arm rather than trusted. It also catches a
# report copied in by hand, which is how the three alpha-surface arms came to
# ship twelve report.json files and no cells at all.
orphans=0
for report in "$DEST"/*report.json; do
  [[ -e "$report" ]] || continue
  stem="${report%.report.json}"
  [[ -f "${stem}.cells.csv" || -f "$DEST/cells.csv" ]] && continue
  log "ORPHAN  $(basename "$report") has no cells beside it"
  orphans=$((orphans + 1))
done
if (( orphans )); then
  log "$DEST is staged on disk and was NOT committed"
  die "REFUSING TO PUBLISH: $orphans report(s) carry no cells.csv. A report is a
[publish] verdict and the cells are the measurement it was read off; published
[publish] apart, the verdict cannot be rechecked and nobody can tell which
[publish] numbers rest on 5 samples and which on 50. Copy the cells in with
[publish] --reports <the directory that holds both>."
fi

if (( SHA_OK == 0 )); then
  if [[ -n "$MISSING_SHA_REASON" ]]; then
    log "OVERRIDDEN with --allow-missing-sha: $MISSING_SHA_REASON"
    log "  the admission is in $DEST/SUMMARY.md"
  elif (( DRY )); then
    log "WOULD REFUSE TO COMMIT: a row names a commit this repository does not"
    log "  have (report in $SHA_REPORT). Rows that cite code nobody can check"
    log "  are not reproducible. Push the commit that made them if it exists,"
    log "  or publish with --allow-missing-sha 'why this is acceptable' and the"
    log "  reason goes into SUMMARY.md where a reader will find it."
  else
    log "$DEST is staged on disk and was NOT committed"
    die "REFUSING TO PUBLISH: a row names a commit this repository does not have
[publish] (report in $SHA_REPORT). Rows that cite code nobody can check are not
[publish] reproducible. Push the commit that made them if it exists, or publish
[publish] with --allow-missing-sha 'why this is acceptable' and the reason goes
[publish] into SUMMARY.md where a reader will find it."
  fi
fi

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
  # Show git's own error rather than guessing. The previous version discarded
  # it and blamed credentials, which sent someone hunting for a token when the
  # actual cause was a non-fast-forward: two pods and a laptop push to this
  # repo, so a diverged branch is the normal state, not an exception.
  if err="$(git push origin HEAD 2>&1)"; then
    log "pushed. The result set is now on GitHub."
    # NOTHING IS APPENDED TO THE ARM HERE. The reachability verdict was taken
    # before the commit, so it is inside the arm that was just pushed; writing
    # to it now would modify a tracked file and dirty the tree behind a publish
    # that succeeded.
  else
    log "push failed. git said:"
    printf '%s\n' "$err" | sed 's/^/[publish]   /'
    if printf '%s' "$err" | grep -qi 'non-fast-forward\|fetch first\|rejected'; then
      log "  the branch has diverged. run:"
      log "    git pull --rebase origin main && git push origin HEAD"
    elif printf '%s' "$err" | grep -qi 'authentication\|could not read\|permission'; then
      log "  no credentials. run: gh auth login"
    fi
    log "  the commit is safe locally either way, but nothing this arm's"
    log "  GIT_SHA_CHECK.txt calls PENDING reached the remote: that verdict was"
    log "  taken on the promise of the push that just failed."
  fi
fi
