# Sourced by main.sh --profile blind (never run on its own), after M0's env-preflight, env, power-readback,
# unlock-start and revision-pinned: ONE blind model per 1x GH200, lean (~1.2 h), scored on the VM.
#
#   bash scripts/prod/main.sh --profile blind --model KEY --utc STAMP     # results prod-<KEY>-<STAMP>
#
# WHAT RUNS (2026-11-20; everything main.sh's full profile adds beyond this is dropped: Adefault, Aon, Beager, replay
# C, the replay offsets, the extras):
#   B0  in the BACKGROUND (kitlib.sh `step --bg` / `step_join`): price-check through v8, v9 AND v10 (CPU), the model's
#       files at its pin (prodlib.py prefetch), the prompts (pinned ShareGPT, the model's tokenizer at its pin) and
#       Frontier's venv (frontier_setup.sh), all while
#   B1  sag1410 (GPU) measures the board's 128x128 par; then every B0 step is joined (a hard FAIL stops the run)
#   B2  the 60 s gate (graphs + nsys node tracing at default clocks: G-CFG, G-REV, ... check-sequences)
#   B3  Frontier's grid-check and the early probe (GPU, hard)
#   B4  A1710 (decode, G-COV on the decode tiles only) and A1410 (prefill, G-COV on the prefill tiles only), each
#       capture in the foreground; each run's nsys export + extract in the BACKGROUND during the next GPU step
#   B5  Frontier's timers (GPU, both branches at their clocks), then fpoints, the CSVs, and the three trainers (cli,
#       simpath, cli-fills) IN PARALLEL in the background (CPU), then Frontier's predictions on both runs
#   B6  the score ON THE VM: offline.sh (v8_ms, v9_ms and v10_ms beside it, the roofline, Frontier's three arms; no replays),
#       nproc workers (cap 32); pushes after the captures (background) and at the end; held-back-check last.
# GPU-timed steps (sag1410, gate60, the probe, the captures, the Frontier timers) never overlap one another; only CPU
# and network work runs beside them.
NW="$(nproc 2>/dev/null || echo 4)"
[[ "$NW" -gt 32 ]] && NW=32

prompts_all() {   # the pinned ShareGPT file, then the prompt set with the model's tokenizer at its pinned revision
  "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data" || return 3
  "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --tokenizer "$TOK" --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt --out "$R/prompts.jsonl"
}
export_extract() {   # export_extract LABEL: the run's nsys export, then its per-call extract (CPU only)
  "$NSYS" export --type sqlite --force-overwrite true -o "$TR/$1/serve.sqlite" "$TR/$1/serve.nsys-rep" || return 3
  "$PY" "$SC/nsys_extract.py" --run "$R/$1" --sqlite "$TR/$1/serve.sqlite" --model "$MODEL"
}
blind_capture() {   # blind_capture LABEL LOCK BRANCH: lock, the capture (G-COV on BRANCH's tiles), unlock
  local lab="$1" lock="$2" br="$3"
  step "$lab-lock" --always --cap 120 -- kit_lock "$lock"
  step "$lab-capture" --cap 10800 --needs "$lab-lock" --out "$TR/$lab/serve.nsys-rep,$R/$lab/manifest.json" -- \
    "${NSYS_A[@]}" -o "$TR/$lab/serve" "${SERVE[@]}" --mode graphs --shared-stream off --cov-branch "$br" --out "$R/$lab"
  step "$lab-unlock-after" --always --cap 60 -- kit_unlock
}
export R TR MODEL TOK SC PY NSYS FR_ROUTING CSVREL PROBE_POINTS

# B0 + B1
step price-check --bg --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model "$MODEL" --pricer "$PRICER"
step price-check-v9 --bg --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model "$MODEL" --pricer r3_timing_model_v9
step price-check-v10 --bg --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model "$MODEL" --pricer r3_timing_model_v10
step model-prefetch --bg --cap 3600 -- "$PY" "$SC/prodlib.py" prefetch --model "$MODEL"
step prompts --bg --cap 3600 --out "$R/prompts.jsonl" -- prompts_all
step frontier-setup --bg --cap 3600 --out "$FR/frontier.env" -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary
step sag1410 --cap 5400 --out "$R/sag/par.json" -- bash "$SC/sag1410.sh" --out "$R/sag"
step unlock-after-sag --always --cap 60 -- kit_unlock
for s in price-check price-check-v9 price-check-v10 model-prefetch prompts; do step_join "$s"; done
step serve-plan --cap 300 --needs prompts -- "${SERVE[@]}" --mode graphs --out "$R/A1710" --dry-run

# B2: the 60 s gate (the weights are already in the HF cache: model-prefetch)
step gate60-capture --cap 3600 --needs model-prefetch --out "$TR/gate60/serve.nsys-rep,$R/gate60/manifest.json" -- \
  "${NSYS_A[@]}" -o "$TR/gate60/serve" "${SERVE[@]}" --mode graphs --shared-stream off --min-steps-per-tile 0 \
  --max-seconds 60 --out "$R/gate60"
step gate60-extract --cap 1800 --needs gate60-capture --out "$R/gate60/calls.jsonl|$R/gate60/calls.shards.json" -- \
  export_extract gate60
step gate60-sequences --cap 600 -- "$PY" "$SC/nsys_extract.py" check-sequences --run "$R/gate60" --model "$MODEL"

# B3: Frontier's venv (joined), its grid-check and the early probe
step_join frontier-setup
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
export FSRC FR_PY_PRIMARY 2>/dev/null || true
step frontier-grid-check --cap 300 --needs frontier-setup -- "$PY" "$SC/frontier_op.py" grid-check --model "$MODEL" \
  --src "$FSRC" --max-tokens 16384
step frontier-timer-probe --cap 900 --needs frontier-grid-check --out "$R/frontier-probe/predictions/index.json" -- fr_probe

# B4: the two primary runs; each run's export + extract in the background during the next GPU step
blind_capture A1710 1710 decode
step A1710-extract --bg --cap 5400 --needs A1710-capture --out "$R/A1710/calls.jsonl|$R/A1710/calls.shards.json" -- \
  export_extract A1710
blind_capture A1410 1410 prefill
step A1410-extract --bg --cap 5400 --needs A1410-capture --out "$R/A1410/calls.jsonl|$R/A1410/calls.shards.json" -- \
  export_extract A1410
step_join A1710-extract

# B5: Frontier's timers (GPU), then its CPU work, the three trainers in parallel
fr_passes frontier-timer "$FR_ROUTING"
step frontier-timer-unlock --always --cap 60 -- kit_unlock
step_join A1410-extract
step push-primary --bg --soft --cap 3600 -- kit_push "prod blind $MODEL: A1710, A1410 and Frontier's timers"
fr_fpoints frontier-points frontier-timer "$FD/points.json"
step frontier-csv --soft --cap 600 --needs frontier-points --out "$CSV" -- "$PY" "$SC/frontier_op.py" csv --model "$MODEL" \
  --points "$FD/points.json" --src "$FSRC" --truth kernel_sum --out "$FD/profile"
step frontier-csv-fills --soft --cap 600 --needs frontier-points --out "$FD/profile-fills/$CSVREL" -- "$PY" \
  "$SC/frontier_op.py" csv --model "$MODEL" --points "$FD/points.json" --src "$FSRC" --truth with_fills \
  --out "$FD/profile-fills"
step frontier-train --bg --soft --cap 1800 --needs frontier-csv --out "$FD/models" -- "$PY" "$SC/frontier_op.py" train \
  --model "$MODEL" --csv "$CSV" --src "$FSRC" --out "$FD/models"
step frontier-simpath --bg --soft --cap 1800 --needs frontier-csv --out "$FD/models-simpath" -- "$PY" \
  "$SC/frontier_op.py" simpath --model "$MODEL" --arm-label simpath --csv "$CSV" --src "$FSRC" --out "$FD/models-simpath"
step frontier-train-fills --bg --soft --cap 1800 --needs frontier-csv-fills --out "$FD/models-fills" -- "$PY" \
  "$SC/frontier_op.py" train --model "$MODEL" --arm-label cli-fills --csv "$FD/profile-fills/$CSVREL" --src "$FSRC" \
  --out "$FD/models-fills"
for s in frontier-train frontier-simpath frontier-train-fills; do step_join "$s"; done
FMODELS=()
for m in cli=models simpath=models-simpath cli-fills=models-fills; do   # a model set counts once ITS trainer passed
  if [[ $KIT_DRY -eq 1 || -f "$FD/${m#*=}/versions.json" ]]; then FMODELS+=(--models "${m%%=*}=$FD/${m#*=}"); fi
done
step frontier-predict --soft --cap 3600 --out "$FD/predictions/index.json" -- "$PY" "$SC/frontier_op.py" predict \
  --model "$MODEL" ${FMODELS[@]+"${FMODELS[@]}"} --csv "$CSV" --calls "A1710=$R/A1710" --calls "A1410=$R/A1410" \
  --src "$FSRC" --out "$FD/predictions"
step_join push-primary

# B6: the score on the VM (offline.sh: expected-outputs reads kit-profile.json: blind), then the pushes
step score --soft --cap 7200 --out "$R/score/prod.score.json" -- env PY_SCORE="$PY" bash "$SC/offline.sh" --results "$R" \
  --model "$MODEL" --pricer "$PRICER" --also-price v9=r3_timing_model_v9 --also-price v10=r3_timing_model_v10 --workers "$NW"
step push-final --soft --cap 3600 -- kit_push "prod blind $MODEL: measured, Frontier predicted, scored on the VM (v8, v9, v10, Frontier)"
step held-back-check --cap 120 -- "$PY" "$SC/prodlib.py" held-back-check \
  --held "${MOE_PUSH_HOME:-$HOME/moe}/push/HELD-BACK.txt" --results "$R"
