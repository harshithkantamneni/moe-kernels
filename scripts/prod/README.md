# Production-check kit (scripts/prod)

The kit measures vLLM 0.27.1's fused MoE call inside a real serving run on a GH200 and scores the frozen model
(v8), a roofline and Frontier against it. The numbers are descriptive: no bars, no verdicts. The spec is
KIT-SPEC.md, with PLAN.md, CAPTURE.md and CRITIQUE-DISPOSITION.md (2026-11-19). Nothing pinned or frozen is
edited. moe/spec.py is not touched: the ERNIE MoEConfig is built in `prodlib.py`.

## Files

| file | role |
|---|---|
| `moe_capture.py` | The hook inside the vLLM process. One Triton kernel per routed call writes topk_ids (int16) and a record (layer, padded n, offset, seq) into a device ring; it launches grid (1, layer + 1), so every marker carries its layer in gridY. The host drains the ring between steps into npz shards. A step log records (step, T, padded T, cudagraph mode). |
| `serve_capture.py` | One offline serving run: closed-loop concurrency levels, graphs or eager, shared-experts stream on/off, optional torch-profiler window. Writes the manifest, the gates and a 10 Hz NVML clock log. Also holds the T-histogram estimator. |
| `prompts.py` | The deterministic prompt set: ShareGPT, with a Gutenberg text as fallback. Each source is pinned by URL and sha256. The set file is tamper-evident (it carries the sha256 of its own body). |
| `nsys_extract.py` | Reads the nsys sqlite export (mode A) or the torch-profiler trace (mode B). Splits the timeline into calls (align..moe_sum, router excluded), by stream and by sequence, and reports kernel-sum and span per call. Runs the gates G-REC, G-GEMM, G-PAIR, G-STEPMARK, G-STEP, G-IDS, G-GRID and G-CLASS. `check-sequences` and `replay-check` are the smoke's kernel-sequence gates. |
| `replay.py` | Picks the registered stratified sample, scaled to a fixed time budget by the smoke's wall times. Re-times each sampled call with r3_decode's GRR instrument on its real ids. Runs under locked_r3. Writes each cell's wall time. |
| `sag1410.sh`, `sag1410.plan`, `sag_par.py` | The board's 128x128 sag par at 1410 MHz. The v7 prefill blind's own driver units and finaliser, unchanged (see "The prefill constants"). |
| `offline.sh` | After the main rental, on the laptop: decompress and check the fetched tree, then the score (Frontier's predictions come from the VM). |
| `hf_revisions.py` | On the laptop: the Hub's current commit of every model and tokenizer repo (public API, no auth), to pin in `prodlib.HF_REVISIONS`; `--check` holds the pins to the Hub. |
| `frontier_setup.sh`, `frontier_op.py` | Frontier at db5a85b in its own venv (vLLM 0.27.1's closure + Frontier, `-e --no-deps`). **Its predictor, our profiles** (owner decision 2026-11-19, see "Frontier"): Frontier's full token grid and its default "uniform" routing (3 seeded samples per point, its own generator ported verbatim), timed by a lightweight timer that mirrors Frontier's profiler; Frontier's CSV schema under two truth definitions, its trainer (CLI), its hold-out and its batch predictions; a partial arm when grid points are missing. |
| `plan_time.py`, `plan_lengths.json` *(extra)* | main.sh's expected wall time per step and per block, from the smoke's measured step seconds scaled to ERNIE (main.sh's dry run prints it). `plan_lengths.json` is the smoke prompt set's (input, output) length mix the serving estimate uses. |
| `replay_wall.json` | The smoke's replay wall times (OLMoE-1B-7B, repeats 3), copied from the smoke rental: main.sh's replay budget. |
| `score_prod.py` | Joins the calls with the v8 prices (at the board's par, and DESCRIPTIVE at the reference pars' median), the registered roofline and Frontier. Reports MAPE, wMAPE, p50/p95 \|e\| and bias, each with step-bootstrap CIs, per tile, branch and stratum, plus per-step totals and the run deltas. |
| `pull_traces.sh` *(extra)* | On the laptop, before the terminate: rsync the run's raw traces (`$MOE_HOME/traces/<run>/` on the VM, never pushed) over ssh with the runs' known_hosts, then check every sha256 against the pushed `traces.sha256.json`. |
| `smoke.sh`, `main.sh` | The VM sequences, fail-fast. Each step prints PASS/FAIL with its reason. `--dry-run` prints the steps (a soft one marked `[soft]`) and runs nothing. |
| `prodlib.py` *(extra)* | Code several scripts share: the models (ERNIE is v8's own `ERNIE`), tiles, branches and strata, the capture sizes, the ring format and decoder, the prefill constants, the registered roofline, the pinned HF revisions, and **`price_v8`, the one v8 binding** (`PRICER_MODULE`, or `--pricer` / `$PROD_PRICER`). `prodlib.py price-check` prices one real decode and one real prefill call through v8. |
| `kitlib.sh` *(extra)* | The step runner (caps, `--needs`, `--out`, `--always`, resume), the kit's TMPDIR and PATH (`kit_env`), the lock and unlock helpers, nsys location, and the env and power checks that both sequences use. |
| `env_preflight.py` *(extra)* | The first hard step of smoke.sh and main.sh (`env-preflight`, cap 600 s): every environment fault the smoke rental met, checked before anything long. See "Smoke rental fixes". |
| `tests/prod_fakes.py`, `tests/test_prod_*.py` | CPU tests on synthetic inputs. |

## How each KIT-SPEC item is met

1. **Capture hook.** It binds through `BaseRouter.set_capture_fn` (router/base_router.py:185; called at
   296-297 with the logical ids, before EPLB). It binds per `MoERunner.layer_id` exactly as vLLM's own
   `_bind_routed_experts_capturer` does (gpu_model_runner.py:7739-7772). The binding happens at the top of a
   wrapped `Worker.compile_or_warm_up_model` (gpu_worker.py:679), so it precedes the warmup runs (706-708) and
   `capture_model` (716-717).

   Graph safety:
   - n, the layer and the strides are runtime arguments marked `do_not_specialize`, so one compiled variant
     serves every n. `warm()` launches every capture size in int32 and int64 before any graph capture, and
     stops the run (G-VARIANTS) unless exactly one variant per ids dtype was compiled.
   - The launch grid is (1, layer + 1) and only program (0, 0) works, so the trace carries each call's layer
     in the marker's gridY (a launch parameter, not a specialisation key).
   - Slot reservation is an atomic on a device counter.
   - The capture fn never syncs. The only syncs are in `drain()`, which refuses to run while a stream is
     capturing.

   Counters reset after `LLM()` returns, and the reset drops an NVTX mark that nsys_extract cuts at. Overflow is
   caught three ways:
   - a step that cannot fit is refused before it runs;
   - a call that does not fit writes nothing and still advances the counters, which the drain catches (G-OVF)
     and which also poisons the shard;
   - the drain is scheduled from the step log's padded T, so the ring never fills.

   The step log comes from a wrapper on `_determine_batch_execution_and_padding` (gpu_model_runner.py:3932;
   called at 4271, padded T at 4289). `_dummy_run` is wrapped to mark dummy steps.
2. **serve_capture.py.** It records the vLLM version, the backend of every bound layer, the engine config
   (capture sizes, cudagraph mode), the environment, and the GPU clocks and power at start and end. The run
   stops (exit 2) if the backend is not TRITON (`unquantized_backend`, unquantized_fused_moe_method.py:53), if
   config.json disagrees with the kit's MoEConfig, or if the capture sizes differ.

   Concurrency design: the default levels are `1,8,24,48,80,112,256,1024x3`, with N 8 per level and quota
   max(N, 2L) (N was 64; the smoke rental's timings put the whole main run over 4 h with it, and N only moves the
   T≤32 tile: the larger levels' quotas are 2L). The estimator (a model of V1's decode-first chunked-prefill
   scheduler; on the smoke it predicted 11917 steps, the smoke logged 11922) predicts, on the smoke prompt set's
   length mix (`plan_lengths.json`), these steps per tile:

   | tile | steps |
   |---|---|
   | T≤32 | 8356 |
   | ≤64 | 1346 |
   | ≤96 | 911 |
   | ≤128 | 588 |
   | ≤512 | 1518 |
   | >512 | 1920 |
   | G16 (≥8256) | 41, all from the three 1024 bursts |

   Every tile is ≥30, which a test asserts. After the run, G-COV checks the realized histogram against the same
   minimum.
3. **prompts.py.** Covered by the prompt tests: a deterministic build, the pin check, and refusal of an edited
   set.
4. **nsys_extract.py.** Alignment: the k-th marker kernel after the reset ↔ the k-th ring record, checked
   call by call by G-PAIR (the marker's gridY must be the record's layer + 1; a pairing shifted by one call
   fails it, which a test shows). There is no NVTX per-call alignment (graph replays carry no per-call range).
   The parts of a call are found by stream and by sequence: only the marker's stream, between two markers,
   in stream order. So a shared-expert SiluAndMul on a second stream (stream on) or before the router (stream
   off) never enters the call; tests show both. More than one fill kernel on the naive path fails G-CLASS
   rather than picking one. Truth is measured two ways: kernel-sum (classified kernels only; an intruder
   such as a shared-expert GEMM on the same stream inside the span is counted and excluded) and span.
   Per-call SM clock comes from the NVML sampler's `clk i` NVTX marks.
5. **replay.py.** It reuses r3_decode's `GPUBackend` (GRR rotating weight copies, a CUDA graph of back-to-back
   calls, `launch_floor.mirror_time_cell`) and its `run_page`, spy and V-ALIGN, GD4/GD5, COPIES, GD1,
   `Store` and `write_report`, all unchanged. **The adapter** (smallest possible, disclosed) does two things:
   - it builds the `DecodePage` straight from the sample, which bypasses `page_from`'s T≤128 check;
   - its `RealIdsMixin.build` passes the call's logged ids instead of `realize_counts`.

   Each branch is timed at its own clock (decode 1710, prefill 1410) through `locked_r3.py --r3-script
   scripts/prod/replay.py`. Every locked_r3 call names its own run cap (`--run-cap-s`, below its step's cap:
   main's `replay-<b>` 5000 s under 5400 s, the probes and the `replay-offset-*` passes 1700 s; the smoke 3300 /
   1700 s): locked_r3's 1800 s default would stop a long branch mid-run. The plan's `session`/`WRITES TO` lines are r3_decode's, so locked_r3 parses them,
   and a test checks this. The sample rule is fixed: an equal quota per tile, systematic within a tile, seed
   20261119; main.sh splits 1500 as 857 decode and 643 prefill.

   **The time budget** (registered rule, `replay.budget_quota`). Every replay writes `cell_wall.json`: each
   cell's wall seconds (its checks, captures and every repeat). The smoke's walls, merged into
   `replay_wall.json`, price main's sample. A tile's cost per cell is 1.5 × the smoke's median wall for that
   tile × (main's repeats + 1) / (smoke's repeats + 1). A tile the smoke did not time takes its branch's
   dearest timed tile. If the registered quotas cost more than `REPLAY_BUDGET_S` (2400 s per branch), every
   quota is scaled by budget / cost, floored, with at least 1 per tile. The sample records the inputs and the
   scaled quotas.

   **The big cell.** The smoke's prefill sample always holds a T ≥ 8256 cell: the run's largest one, or else a
   synthetic T 16384 cell (seeded uniform routing, label `s16384`, never scored). The smoke's prefill replay is
   a hard step. **GRR memory** is bounded: every call's output stays live in the graph, so calls × T_max × H ×
   2 bytes is added to r3_decode's single-call plan, and a page over the free memory or over 8 GiB of outputs
   is refused.

   The replay also runs under nsys node tracing, per branch at its branch clock under locked_r3 like replay C
   (SMOKE-FIXES 7; main: soft, `replay-offset-*`; smoke: hard), with `--cell-marks` (the step-marker kernel,
   gridY = the cell's index + 1, before each build of a cell, outside every timed region), so each cell's calls are
   its own. `nsys_extract.py replay-check` then requires each sampled cell's launch grids among its calls and a
   CORE kernel sequence (align..moe_sum, the API path's fill kernels stripped: SMOKE-FIXES 6) that serving showed on
   the same tile, and records per call, for the replay and for the serving call, the fill kernels' count and summed
   GPU time (`replay_check.json`).
6. **Frontier: its predictor, our profiles** (owner decision 2026-11-19). Frontier's MoE profiler cannot run on the
   GH200 (smoke logs `frontier-profile-*`): on vLLM 0.27.1 it builds the gate with `ReplicatedLinear`, which needs
   vLLM's tensor-parallel group and `set_current_vllm_config` ("tensor model parallel group is not initialized");
   its own pinned stack (torch 2.8 + vLLM 0.10.2) has no CUDA torch for aarch64 (the cu128 index has no aarch64
   wheel; PyPI's aarch64 torch 2.8 is CPU-only). So:
   - **Profiler: a lightweight timer that mirrors Frontier's method** (`frontier_op.py timer`; Frontier's own
     profiler cannot run on the GH200, see above). Everything else is Frontier's. Frontier's
     `profile_fused_moe_kernel` builds the weights once and per point calls vLLM's functional `fused_experts`
     eagerly in one process on those warm weights: 2 warm-up calls, a synchronize, then 20 timed calls each
     followed by a synchronize, the row's time the median (`moe_wrapper.py` WARMUP_STEPS / ACTIVE_STEPS). The timer
     does the same, one process per branch pass: ERNIE's weights built once (synthetic bf16, the served shapes), per
     grid point and sample the routing (Frontier's profiling default, "uniform", see below) and the call replay C makes
     (vLLM 0.27.1's functional `fused_experts`, align..moe_sum: the same functional-vs-modular caveat), 2 warm-up
     calls, then a marker kernel (gridY = point index + 1) and 20 timed calls. A sink marker (gridY 65535) before
     each point's routing build and warm-up keeps them out of every point's segment. The time is **the kit's
     kernel-sum**, not a CUDA event: per timed call the core kernels' GPU durations summed (the API fills stripped
     and reported separately, the same classifier as `nsys_extract`), the row's median over the 20, in ms. The loop
     runs under nsys (eager, no graph tracing) at the branch clock, locked_r3-equivalent: decode points (T≤128)
     under `kit_lock 1710`, prefill points under `kit_lock 1410` (two passes), the lock read back by the timer
     (NVML SM clock after the first warm-up; off by more than one 15 MHz step refuses the pass) and the SM clock
     recorded per point before and after its timed calls. A point off the lock is flagged (`off_lock` in
     `points.json`, counted by `frontier-points`), not dropped. The raw trace (`timer.nsys-rep`, `timer.sqlite`)
     is written under `$MOE_HOME/traces/<run>/frontier/`, outside the pushed tree (`fpoints --timer-sqlite`).
   - **Schema, grid, routing, trainer, predictor: Frontier's.** The grid is `get_num_tokens_to_profile(16384)`
     (387 points; `grid-check` holds `frontier_op.token_grid` to Frontier's own function on the VM), **all of it**
     (owner decision 2026-11-19: a budgeted subset would starve Frontier's random forest and bias the comparison
     against it); `frontier-points` fails when any grid point (or one of its samples) is missing. **The routing is
     Frontier's grouped-GEMM profiling default** (review 2026-11-19 M1): `frontier/profiling/moe/main.py:282-295`
     enables load imbalance with `--load_distributions ["uniform"]` and `--num_samples_per_distribution 3`, one test
     case per (num_tokens, distribution, seed 0..2). The timer runs that generator, `load_distribution.py
     generate_expert_routing("uniform")`, ported verbatim (`frontier_op.uniform_routing`: `torch.manual_seed(seed)`,
     `rand(T, E)` on the card, argsort, the first k ids; float16 weights `rand(T, k)` normalised, cast to float32
     for the functional API), three seeded samples per grid point, one CSV row each (1161 rows). A test pins it to
     the values Frontier's own function draws. The round robin (`uniform_topk`: token t takes experts (t·k + j)
     mod E; `balanced` in load_distribution.py) is NOT Frontier's grouped-GEMM default: it is only the gating
     operator's `--routing_runtime_path` (main.py's default is `standard_fused_topk`; `profile_moe.sh` passes
     `uniform_topk`), whose metadata columns the rows carry for that untimed operator. The round robin is
     available as a labelled second arm, off by default (`main.sh --frontier-balanced-arm`, arm `cli-balanced`;
     `PROD_FRONTIER_ROUTING` sets the primary arm's routing). The CSV is
     the row Frontier's profiler writes for the grouped GEMM (`moe_kernel_only.csv`): num_tokens, the 14
     load-imbalance features from Frontier's own `MoELoadImbalanceInput(expert_token_counts=...)`, the model and
     routing columns, `measurement_type KERNEL_ONLY`, `profiling_precision BF16`, and
     `time_stats.moe_grouped_gemm.{median, mean, min, max, std, count, kernel_count, num_tokens_post_padded,
     block_size_m, block_size_m_ranges}`. The three operators we do not time (gating linear, routing top-k,
     shuffling) have their target columns **empty**. The trainer is `python -m frontier.training.cli moe`
     (explicit parameters, KERNEL_ONLY: its parser, MoETrainer, random-forest grid search, K-fold CV, pickled
     estimator) in Frontier's venv, with **one shim**: `MoETrainer._get_model_names` is narrowed to
     `["moe_grouped_gemm"]` (the CLI would stop on the empty targets). `simpath` is the same trainer with the
     simulator's feature selection (total_routed_tokens, num_tokens_post_padded). The hold-out is Frontier's
     fidelity on its own profiled points (every 5th num_tokens held out). The predictions run **on the VM**
     (`frontier-predict`, last): every logged call of every run, one DataFrame per model, written compact
     (`frontier/predictions/`: `index.json`, `<run>.g.npy`, `<arm>.<run>.npy`).
   - **Two truth definitions** (review m4), two CSVs, two trained models, both scored against serving's kernel-sum
     and labelled: `cli` is trained on the kernel-sum WITHOUT the functional API's fills (align..moe_sum: the same
     definition as the scorer's serving truth, since serving's modular `TritonExperts` launches no fills): **the
     like-for-like arm**. `cli-fills` is trained on Frontier's NATIVE definition: its `record_function` scope sums
     every kernel the functional call launches, the ~9 fills included (`csv --truth with_fills`); it carries the
     fills as error by construction. The same caveat applies to v8: v7/v8's decode constants were calibrated on the
     functional API, so they include the fills (SMOKE-FIXES 6).
   - **A missing grid point** (review M3): `fpoints` still writes `points.json` and FAILS; the primary CSV refuses
     an incomplete grid, so `cli` is not trained. `frontier-csv-partial` (`csv --partial`) then writes the
     separately labelled PARTIAL arm with `profile-partial/csv_meta.json` listing the missing points (and samples),
     and `frontier-train-partial` (`--only-if-partial`) trains it as `cli-partial`, soft and secondary. On a complete
     grid both steps PASS and write no model.
   - **The early probe** (hard, right after the 60 s gate, ~6 min): Frontier's venv and grid-check, then
     `frontier-timer-probe` runs the new timer on 3 ERNIE grid points (64; 1024, 8192: one decode T, two prefill T,
     `frontier_op.PROBE_POINTS`, 3 samples each) at their branch clocks, then `fpoints`, the CSV, Frontier's own
     trainer with a narrow grid search (`train --narrow`: the CLI's own `--k_fold_cv_splits 3 --num_estimators 50
     --max_depth 8 --min_samples_split 2`) and its predictions on those points (`predict --points`, refused on a
     non-finite or non-positive prediction). A failure stops the run before the expensive blocks. The smoke runs
     the same probe on 64, 192, 256.
   - **No budget.** The whole grid with its 3 samples per point costs about 6 min of GPU time (two timer passes and
     their exports, `plan_time.py`), so every point is measured; there are no pages.
   - Checked on the laptop against Frontier db5a85b's own code (a scratch copy of the clone, CPU only, synthetic
     times on all 387 points of a synthetic timer trace): the CSV loads, the CLI trainer trains `moe_grouped_gemm` on
     the 14 features, simpath on the 2 padded features, the hold-out and the batch predictions run, and the grid
     equals `get_num_tokens_to_profile(16384)`. An earlier run of that check found and fixed a simpath bug
     (MoETrainer calls `_get_training_df_for_model` by keyword).
7. **score_prod.py.** Descriptive only, and run on the laptop after the rental (`offline.sh`). Kernel-sum is the
   primary truth and span the secondary one: calls with T>512 run without CUDA graphs, so their span includes the
   host's launch gaps (the smoke's 128x128 span was ~2x kernel-sum); each run reports span / kernel-sum per
   cudagraph mode. With `--replay-check BRANCH=replay_check.json` the replay-vs-serving delta is reported with and
   without the API path's fills (replay_ms minus the cell's fills), with the fills' share of in-serving kernel-sum
   and of v8 per tile. v7/v8's decode constants were calibrated on the functional API (fills inside T0 and align),
   so v8 is expected to over-predict in serving by about the fills' time. Pricing runs in
   a process pool (4 workers by default), cached by (T, counts digest, ids digest, par). Every exception a
   worker meets is recorded as that job's error, `BaseException` included (v8's PoisonedTread), and fails the
   score. Predictors:
   - `v8_ms`: v8 at the board's par (`--sag`, required).
   - `v8_refpar_ms`: DESCRIPTIVE, v8 with no on-board calibration, at the median of the four registered
     reference pars.
   - `roof_ms`: the roofline, always scored (see "The roofline").
   - Frontier, one predictor per arm (`frontier:<arm>`), each labelled in the report (`score_prod.FRONTIER_ARMS`):
     `cli` (the like-for-like, fills-free truth), `simpath`, `cli-fills` (Frontier's native truth with the fills),
     `cli-partial` and `cli-balanced` when they exist.

   A locked run scores only its own branch. A default-clock run is labelled a production delta only. Off-lock
   calls (more than 15 MHz from the lock) are dropped and counted. The refused-call share of MoE time is
   reported. The refused window is per model: T from ⌈6·E·128/k⌉ to 129·E − 1 (ERNIE 8192..8255).
8–9. **smoke.sh and main.sh.** They follow CRITIQUE-DISPOSITION RENTALS (1) and (2). main.sh runs, in order: the
   env-preflight, the ERNIE 60 s gate, the early Frontier probe (its venv, the grid-check, `frontier-timer-probe`:
   all hard), the sag par, A1710, A1410, replay C, Frontier (the two timer passes over the full grid, the points,
   the CSV, train, hold-out, simpath, the with-fills CSV and model, the partial arm), the secondary runs (Adefault, Aon, Beager on
   one level, the replay under nsys per branch with its replay-check), and Frontier's batch predictions of every
   extracted run. Every secondary and Frontier step is soft. A trap unlocks the card on any exit, and a soft push
   to the run's branch follows each block. The score runs on the laptop (`offline.sh`). Tests assert the order, the
   soft and hard steps, that main.sh has no score step, and that every cap is ≥2x its expected time.

## Review 2 (2026-11-19): what changed

- **Nothing the laptop needs is held back by the push** (MAJOR A). `scripts/vm_results_push.sh` is registered and
  pinned, so it is not edited: it still compresses or holds back any file over 95 MB. The kit writes its per-call
  outputs in shards of at most `prodlib.SHARD_MAX_BYTES` (64 MB): `calls.jsonl` / `counts.npy` become
  `calls.00000.jsonl` ... and `counts.00000.npy` ... with `calls.shards.json` (rows and sha256 per shard) when a
  run is big. `prodlib.read_calls` / `load_counts` read them back (and any `.xz` form), and refuse a missing or
  changed shard. The ring's own shards stay under 76 MB by construction. main.sh's last step,
  `held-back-check` (hard), fails when the push's `SHA256SUMS` holds no file of the run, or when `HELD-BACK.txt`
  lists any file of the run but the rental's intermediates (`*.nsys-rep`, `*.sqlite`, `data/`, Triton caches). The EXIT trap pushes once more with `--final <exit code>`
  whatever stopped the sequence. offline.sh first writes every `.xz` out (`decompress`) and then refuses a tree
  that lacks a run, a model set, the par or a replay that the rental's own `SUMMARY.txt` says it produced
  (`expected-outputs`, hard).
- **The replay budget is the target model's** (MAJOR B). `replay_wall.json` names its model (`wall-merge
  --model`). A wall from another model is scaled by `prodlib.expert_bytes` (one expert's w13 + w2 bf16 bytes:
  OLMoE 12.6 MB, ERNIE 23.6 MB), and a wall without a model is refused. main.sh replays a probe first, per branch:
  `replay.PROBE_PER_TILE` (2) cells per tile. The probe's own ERNIE walls size the rest (`sample --probe-wall`),
  which gets `REPLAY_BUDGET_S` minus the probe's total wall, so the branch stays inside its fixed 2400 s.
- **Every step has a registered wall cap** (MAJOR C). `step NAME [--soft] --cap SECONDS [--needs PRE] -- CMD`
  runs CMD under `timeout --kill-after=60 SECONDS`. A step past its cap FAILS (or is dropped, if soft), so a hung
  soft step cannot hold the rental. A step without `--cap` is refused. A step whose `--needs` prerequisite failed
  is recorded `SKIPPED` and not run: no capture runs after a failed lock (MINOR 6).
- **A whole-step pairing shift fails** (the shift-by-one-step blind spot). The step-log wrapper launches a step
  marker kernel before each step's forward, with grid (1, step % 65535 + 1). G-STEPMARK holds the markers to the
  step log, and each call to the step its ring record joins, n_layers calls per step. A trace shifted by a whole
  step keeps every layer in place, so G-PAIR passes, but G-STEPMARK fails.
- MINOR 1: `v8.DECLARED_REFUSALS` lists each Refused of v8's price path by its own message pattern; the
  `Refused: ` catch-all is gone, so an undeclared refusal fails the sweep (a test fabricates one).
  MINOR 2: G-REV fails a pinned run whose snapshot cannot be resolved. MINOR 3: `hf_revisions.py` resolved both
  repos (the tokenizers are the models' repos), and the shas are in `prodlib.HF_REVISIONS`. MINOR 4: the sag par
  needs the driver's exit 0 and a gate record written after the step began (`sag_par.py --started`). MINOR 5:
  sag1410.sh fails before the driver without `$SESSION_ROOT/census.json` (PF6). MINOR 7: offline.sh writes its
  summary and logs under `<results>/offline-logs/` and creates them itself. MINOR 8: the compiled-variant counts
  are read again at `close()`, and G-VARIANTS-END fails the run if they changed after warm.

## Smoke rental fixes (2026-11-19)

The smoke rental on a GH200 met five environment faults (SMOKE-FIXES.md). Each is fixed in the kit, and each now
fails the first step instead of a step deep in the sequence.

1. **Power.** `kit_power_check` reads `enforced.power.limit`. The board reports `power.limit` 900 W (the default
   setting) and enforces 700 W.
2. **nsys temp dir.** After setup_vm's ncu, `/tmp/nvidia` is root-owned, so nsys could not create
   `/tmp/nvidia/nsight_systems`. `kit_env` creates `$MOE_HOME/tmp` (`$KIT_TMPDIR` overrides it) and exports it as
   `TMPDIR` for every step, so every `nsys profile` and `nsys export` uses it. `--dry-run` prints it (`ENV TMPDIR=`).
3. **PATH.** The kit runs the venv's python by absolute path, and FlashInfer's JIT (vLLM's sampler) runs `ninja`.
   `kit_env` puts the venv's bin dir (`dirname $PY_VLLM`) first on PATH for every step (`ENV PATH first=`). A bare
   `python3` in a step (locked_r3.py) therefore runs the venv's python3; locked_r3 imports only exit_codes.
4. **G-REV resolver.** vLLM downloads only the files it needs, so `snapshot_download(local_files_only=True)` raised
   IncompleteSnapshotError. `serve_capture.resolve_revision` now takes the snapshot directory that config.json
   resolves to (`huggingface_hub.try_to_load_from_cache`), and requires the weights (the safetensors index, else
   the single weight file) to resolve to the same one.
5. **Dirty tree.** The pinned v7p driver refuses a dirty checkout. `env-preflight` P1 fails a dirty tree, or a HEAD
   other than the expected commit, before anything runs. No kit file is patched on a rental.

Round 2 (the rest of SMOKE-FIXES.md; item 8 was an operator error, not the kit):

6. **Replay-check.** The replay (functional `fused_experts`) launches ~9 `fill:vectorized_elementwise_kernel` per
   call before align; serving (modular `TritonExperts`) launches none, so the full sequences differed and the
   smoke's replay-check failed though the core sequence align → GEMM → act → GEMM → moe_sum was identical.
   `replay-check` now compares the core sequence (`nsys_extract.split_pre_run`: with an align kernel the core is
   serving's own align rule; on the naive path the last fill is the align) and the grids, and records per call the
   fills' count and summed GPU time for the replay and for serving (`replay_check.json`; serving's extraction
   records `fills_n` / `fills_ns` per call too). score_prod reports the replay-vs-serving delta with and without
   the fills (item 7 above).
7. **The nsys replays at the branch clock.** `replay-nsys-{decode,prefill}` ran unlocked, and the prefill pass
   failed G1_lock_thermal (exit 3). They now run under locked_r3 at 1710 / 1410, like replay C, so the instrument
   offset is measured at replay C's clock (smoke `replay-nsys-*`, main `replay-offset-nsys-*`).
9. **frontier_setup.sh is idempotent.** A checkout already at db5a85b is reused (no clone, no fetch); a checkout at
   another commit, or a non-git directory in the way, is refused (exit 4). A test runs both.

Pins from the smoke: the ShareGPT sha256 (`35f0e213...6479ba4`) is in `prompts.py`, and the smoke's
`replay_wall.json` (olmoe-1b-7b, repeats 3; the ERNIE bytes scaling applies) is `scripts/prod/replay_wall.json`.

**`env-preflight`** (`scripts/prod/env_preflight.py`, hard, cap 600 s, the first step of smoke.sh and main.sh)
runs every check and prints one PASS or FAIL line per check, with its reason. It writes `$R/env-preflight.json`.
- P1: clean checkout at the expected commit (`--expect-commit`, else `$PROD_EXPECT_COMMIT`, else the commit in
  `$MOE_HOME/env.sh`'s header).
- P2: no compute process on the GPU.
- P3: the enforced power limit is 700 W.
- P4: 1710 and 1410 are each set by `-lgc` and restored by `-rgc` (skipped when P2 fails).
- P5: TMPDIR is writable.
- P6: ninja, nsys and ncu are on PATH.
- P7: `nsys profile --cuda-graph-trace=node /bin/true` runs under that TMPDIR.
- P8: the model's pinned config.json and weight index download, and the fixed resolver returns the pin.
- P9: Frontier's repo is reachable at frontier_setup.sh's pinned commit.
- P10: more than 200 GB free (`$PROD_MIN_FREE_GB`).
- P11: setup_vm's `$SESSION_ROOT/PREFLIGHT.txt` says `verdict READY`.
- P12: the weight download is priced by a measured rate: up to 1 GB or 20 s of the model's first weight shard at
  its pin is streamed and discarded; under 50 MB/s (`--min-mbps`) it FAILS with the rate, and the detail gives the
  whole download's expected time from the index's `total_size` (ERNIE: 43.7 GB, about 15 min at the floor).

**Raw traces** (review m6). main.sh writes every `*.nsys-rep` and `*.sqlite` under `$MOE_HOME/traces/<run>/`
(gate60, A1710, A1410, Adefault, Aon, graphmode, replay-offset-*, frontier/, frontier-probe/), outside the pushed
results tree, so `vm_results_push.sh` (pinned, not edited) never spends a push xz-compressing multi-GB traces. Each
push first records their sha256 in `<results>/traces.sha256.json` (`kit_push` runs `prodlib.py trace-manifest`; an
unchanged file keeps its hash). Before the terminate, on the laptop: `bash scripts/prod/pull_traces.sh <ip> <run>`
(rsync over ssh with `$MOE_VM_STATE_ROOT/known_hosts`, StrictHostKeyChecking=yes; then every sha256 checked against
the manifest). The smoke does not push, so its traces stay in its results tree.

**Resume.** `smoke.sh`, `main.sh` and `offline.sh` take `--results DIR --from STEP`. The run re-enters at STEP in
the existing DIR, appending to its SUMMARY.txt.
- Every step before STEP is recorded `SKIPPED-RESUME <step>` and is not run.
- A step declares the outputs a later step reads (`--out`). A skipped hard step whose output is missing refuses
  the resume (exit 2, before anything runs past it). A skipped soft step whose output is missing stays dropped,
  and the steps that `--needs` it are SKIPPED.
- The `--always` steps run again: env-preflight, env, power, revision and replay-wall checks, and every lock and
  unlock. So a resumed capture runs at its own clock.
- A STEP that is not in the script's plan, or `--from` without `--results`, is refused.
- `--dry-run` with `--from` checks the outputs and prints the resume plan.

## main.sh: every step, its cap, its expected time

The expected seconds are the smoke rental's MEASURED step seconds (SUMMARY.txt, OLMoE-1B-7B on the GH200,
2026-11-19) scaled to ERNIE where the work scales: expert bytes ×1.875, MoE layers 27/16 (serving work and engine
start ×3.16; replay cells ×1.875, the replay budget's own rule; nsys export and extraction by the call count), a
locked run's serving ×1980/lock. The serving time is serve_capture's estimator on the smoke's length mix, each step
priced by the smoke's median step time in its T band (this prices the smoke's own run at its measured 44 s).
`plan_time.py` computes every row; `main.sh --dry-run` prints them and the total. Steps the smoke did not time are
marked *assumed*. A cap is the step's registered kill time, at least 2x its expected time (a test holds every step to
it). Per block (expected / sum of caps):

| block | what | expected | caps |
|---|---|---|---|
| M0 | env-preflight (*assumed* 150 s, P12's timed download included), env, power, unlock, revision, replay wall, price-check (81 s) | 4.0 min | 34 min |
| M1 | prompts (fetch 3 s, build 9 s x 4) | 0.7 min | 65 min |
| M2 | the 60 s gate: engine start 52 s x 3.16 + 60 s + the weight download (*assumed* 218 s at 200 MB/s; P12 measured), export, extract, sequences; the early Frontier probe: venv (*assumed* 300 s), grid-check, `frontier-timer-probe` 355 s; push | 21.1 min | 270 min |
| M3 | sag1410 (1048 s, the same pages as the smoke) | 17.5 min | 91 min |
| M4 | A1710: capture 678 s (14680 steps), export 25 s, extract 106 s | 13.5 min | 333 min |
| M5 | A1410: capture 788 s, export, extract, push (*assumed* 120 s) | 17.3 min | 393 min |
| M6 | replay C: decode probe 277 s + rest 1440 s, prefill probe 643 s + rest 1196 s (the registered budget, walls right) | 60.5 min | 345 min |
| M7 | Frontier: the timer on all 387 points x 3 samples (decode 123 s, prefill 211 s, exports 10 s each), points, CSV, train / hold-out / simpath (*assumed* 180 s each), the with-fills CSV and trainer (180 s), the partial arm (no model on a complete grid), push | 20.8 min | 455 min |
| M8 | Adefault 741 s, Aon 811 s, Beager 283 s, replay under nsys 217 + 413 s with export and check, Frontier predict (*assumed* 600 s), push, held-back check | 53.8 min | 1048 min |
| | **total** | **12557 s = 3.49 h** (3.67 h if the download runs at P12's 50 MB/s floor) | 50.6 h |

The long steps:

| block | step | hard/soft | cap | expected | basis |
|---|---|---|---|---|---|
| M2 | gate60-capture | hard | 3600 s | 443 s | smoke engine start x 3.16 + 60 s + download (*assumed* 200 MB/s; P12 measures it) |
| M2 | frontier-timer-probe | hard | 900 s | 355 s | two timer passes on 3 points x 3 samples (start 120 s each, *assumed*) + Frontier's fpoints / csv / narrow train / predict (*assumed* 90 s) |
| M3 | sag1410 | hard | 5400 s | 1048 s | smoke |
| M4 | A1710-capture, -export, -extract | hard | 10800 / 5400 / 3600 s | 678 / 25 / 106 s | estimator x smoke step times x 3.16 x 1.158; smoke x calls 2.08 |
| M5 | A1410-capture, -export, -extract | hard | 10800 / 5400 / 3600 s | 788 / 25 / 106 s | as M4, x 1.404 |
| M6 | replay-probe-{decode,prefill} | hard | 1800 s | 277 / 643 s | 2 cells/tile x smoke wall x 1.875 x (9+1)/(3+1) |
| M6 | replay-{decode,prefill} | hard | 5400 s (locked_r3 run cap 5000 s) | 1440 / 1196 s | (2400 s − probe) / 1.5 |
| M7 | frontier-timer-{decode,prefill} | soft | 3600 s | 123 / 211 s | start 120 s (*assumed*: vLLM import, weights once, Triton compiles) + 19 / 368 points x 3 samples x (2 + 20) calls x max(eager host 1 ms (*assumed*), the smoke's replay C call time x 1.875) + 30 ms per sample (*assumed*) |
| M8 | Adefault-capture, Aon-capture | soft | 10800 s | 608 / 678 s | as M4 at default clocks / 1710 |
| M8 | Beager-capture | soft | 1800 s | 278 s | smoke B (one window) x 3.16 |
| M8 | replay-offset-nsys-{decode,prefill} | soft | 2400 s | 217 / 413 s | the sample's cells x the smoke's repeats-1 nsys walls x 1.875 |
| M8 | frontier-predict | soft | 3600 s | 600 s | *assumed* |
| opt-in | `--graphmode-overhead`: graphmode-unlock, graphmode-capture | soft | 60 / 3600 s | 10 min | (the smoke measured it) |
| opt-in | `--frontier-balanced-arm`: two more timer passes (round robin, one sample per point), points, CSV, train | soft | as M7's | ~8 min | as M7, a third of the samples |

The caps only bound a hung step. The primary measurements (M2 to M6) do not move with `--graphmode-overhead` (a
test holds every other step identical with and without it). `PROD_REQUESTS_PER_LEVEL` and `PROD_LEVELS` override
the plan's N and levels.

## The prefill constants (v8 has no defaults)

v8's prefill branch needs three constants per call: the 128x128 sag par (`sag128`), the 64x128 par
(`sag64`) and the K1 knee (`knee`). v8 has no default for any of them; a prefill call missing one raises
ValueError.
- `sag64` = (30.365, 300.0), the v7 prefill blind's registered `v7pcommon.A2_PAR_64`.
- `knee` = (6.0, 2.75), round 4's SEEN K1 (`round5_chain.K1_SEEN`).

Both live in `prodlib.PREFILL_REGISTERED`, and a test holds each to its source. `sag128` is **measured on the
board at 1410 MHz**.

How main.sh (and smoke.sh) measure it, step `sag1410`:
1. `bash scripts/prod/sag1410.sh --out $R/sag` runs `scripts/gh200_v7p_session.sh --plan
   scripts/prod/sag1410.plan --deadline <now + 75 min> --no-push`. The driver is the v7 prefill blind's, not
   copied or edited. The plan's four lines are that blind's own units, verbatim (a test checks them against
   `scripts/plans/rentalV7P-prefill-blind-2026-11.plan`):
   - the prelude: 1410 lock, 700 W read back;
   - the OLMoE-1B-7B f128 bytes page (128x128x64 w8 s3, G 64);
   - the Mixtral-8x7B f128 bytes page;
   - gate `b3sag`.

   The pages land under `$RESULTS_ROOT`, and the gate record at
   `$SESSION_ROOT/gh200-driver/gate-b3sag-1410.json`.
2. `scripts/prod/sag_par.py` runs `v7pcommon.finalise_pars`, unchanged, on those pages and that record. It
   writes `$R/sag/par.json` with schema `moe-kernels/prod-sag-par/1`, verdict PASS or FAIL, the par
   (A µs, τ µs), and the per-page fits. A FAIL is a hard stop. On the v7p rental's own published pages, this
   reproduces that blind's finalised par (64.325, 612.5); a test checks it.
3. Offline, `score_prod.py --sag $R/sag/par.json` passes the par to `prodlib.load_board_par`, which refuses a
   FAIL par. From there the par flows through `prodlib.price_v8(sag128=par)`, then `prodlib.prefill_constants`,
   then `v8.price({..., "sag128": par, "sag64": A2_PAR_64, "knee": K1_SEEN})`, and finally v7's prefill branch
   (round5_chain.price_128 / k5.cell_ms).

M0's `price-check` runs before the par exists, so it prices at the reference median.

## The roofline

The roofline is the v7 decode blind's registered rival: `T0_GR + A·W_e / bw`, with A the active experts and
W_e = 3·H·I·2 bytes (`finalise_decode.py:141`, `r3_timing_model_v4d.roofline_ms`).
- T0_GR is the frozen CAL2 constant, `predictions.frozen.constants.T0_GR_ms` of
  `docs/registered/2026-11-17-v7-decode-blind-gh200.json`.
- bw is v4d's `CARD["bw_gbps"]`.

It is implemented once, in `prodlib.roofline_ms`, and applied to every call. A test reproduces the
registered roofline of all 143 blind cells, and of every row of `scripts/scoring/v7blind/blind.score.json`,
bit for bit.

## Disclosures (Frontier and the replay)

- Frontier's code is not edited: frontier_setup.sh checks `git diff --quiet` before and after setup. Frontier's
  profiler is NOT used (it cannot run on the GH200, see "How each KIT-SPEC item is met" 6): the profiles are ours, in
  Frontier's schema, on Frontier's grid with Frontier's default routing ("uniform", 3 seeded samples per point, its
  generator ported verbatim); the trainer, the hold-out and the predictor are Frontier's.
- **Frontier's truth vs ours** (review m4): Frontier's native target (record_function) includes the functional
  API's fill kernels; serving has none. The like-for-like arm is `cli` (trained without the fills); `cli-fills`
  is Frontier's native definition, reported beside it. v8's decode constants also include the fills (they were
  calibrated on the functional API, SMOKE-FIXES 6), so v8 over-predicts serving by about the fills' time.
- **The shims** (`frontier_op.py`): the CLI trainer runs with `MoETrainer._get_model_names` narrowed to
  `["moe_grouped_gemm"]` (the other three MoE operators are not timed; their CSV targets are empty, never made up);
  `simpath` swaps MoETrainer's feature selection for the simulator's `select_moe_operator_features`
  (total_routed_tokens, num_tokens_post_padded), with BM from Frontier's own `block_size_m_for_tokens` over the
  table's ranges at prediction time (CRITIQUE M1). The documented CLI model is reported separately (`cli`).
- The Frontier venv is the study's vLLM closure (`requirements/resolved-vllm.txt`, with the editable line dropped
  as setup_runpod.sh does) plus Frontier's dependencies, constrained by that closure. Frontier is installed
  `-e --no-deps`. The fidelity venv (Frontier's pinned stack) stays in frontier_setup.sh but no sequence asks for it.
- The points cover the whole grid (387 points x 3 samples); a missing point is recorded in points.json and only
  the separately labelled partial arm is trained then.
- Replay differences from serving: random hidden states, and private cold weight copies (serving's weights may
  be warm in L2 at small T). The ids are the real ones, including padded rows.
- The capture marker kernel runs between the router and align. It is outside both truths, but it does add a
  small cost to every serving step.

## Before the kit freezes (after the smoke rental)

- DONE (2026-11-19): the ShareGPT sha256 the smoke printed is pinned in `prompts.py SOURCES["sharegpt"]["sha256"]`.
- The HF commits are pinned in `prodlib.HF_REVISIONS` (resolved 2026-11-19 by `python scripts/prod/hf_revisions.py`
  on the laptop: the Hub's public model API, no token, nothing about the user sent). Re-run it with `--check`
  just before the freeze; it exits 3 if a repo moved. The pinned commit is passed as the model and tokenizer
  revision to the engine (`LLM(revision=, tokenizer_revision=)`) and to the prompt builder's tokenizer; G-REV fails
  a run whose loaded snapshot differs or cannot be resolved.
- DONE (2026-11-19): the smoke's `replay_wall.json` is `scripts/prod/replay_wall.json`. main.sh's replay budget reads
  it, and main.sh stops at `replay-wall-present` without it.
- v8 is bound: `price_v8` calls `r3_timing_model_v8.price` with v8's own interface, and `price-check` (smoke
  and M0) prices a real decode call and a real prefill call through it.
- **Launcher:** the `vm_run*.sh` `--driver` allow-lists do not name `scripts/prod/*.sh`. Either a `vm_run_prod.sh`
  copy is needed (not built here, because the existing launchers are not edited) or the sequences are started
  over ssh after `setup_vm.sh`. Lambda launch and terminate stay with `~/.lambda/status` / `~/.lambda/rent`.
- `PROD_POWER_W` (default 700) is the enforced power limit that env-preflight and main.sh's power-readback require.
- `PROD_EXPECT_COMMIT` (or `$MOE_HOME/env.sh`'s header) is the commit env-preflight holds the checkout to.

## Assumptions that cannot be verified without the GPU (each is checked in the smoke rental)

| # | assumption | how smoke.sh verifies it |
|---|---|---|
| A1 | The router call runs inside the opaque MoE custom op, so a capture fn bound after torch.compile's profile run still runs, in both eager and graph replay. | `extract-gate-A`: G-REC (ring records == marker kernels) and G-STEP (n_layers records per logged step, n == T_pad). |
| A2 | Triton's scalar `tl.atomic_add` executes once per program and returns the old value; `if ok:` on a scalar compiles. | The same G-REC/G-STEP, plus `decode_shards`' seq/offset checks on every shard. |
| A3 | `do_not_specialize` yields one compiled variant for every n, so no JIT compile falls inside a capture. | `warm()` gates `kernel_variants` == 1 per ids dtype (G-VARIANTS) before any capture. |
| A4 | The unquantized method is modular (`is_monolithic` False) and the backend on SM90 is TRITON (`TritonExperts`). | `serve_capture` G-BACKEND; `bind` refuses a monolithic method. |
| A5 | The per-call kernel sequence is align (small-batch / align+sort ± memset / naive fill), fused_moe_kernel, act_and_mul, fused_moe_kernel, moe_sum, with no chunking (envs.py v0.27.1 has no VLLM_FUSED_MOE_CHUNK_SIZE). | G-CLASS, G-GEMM (2 × records); `sequences.json` lists the 12 most common name sequences, for a human check. |
| A6 | nsys `--cuda-graph-trace=node` works on aarch64 GH200 with driver 580 and reports graph-node kernels with their gridX and gridY on the launch stream. | G-REC/G-PAIR/G-GRID on run A; `gates.json marker_streams`. If it fails, mode B (eager, torch profiler) is the fallback. |
| A20 | Triton compiles the marker's `if tl.program_id(1) == 0:` body, and a grid of (1, layer + 1) reuses the one compiled variant. | G-VARIANTS after warm (warm launches gridY 1, serving up to 28), and G-PAIR. |
| A21 | The step marker launched from the step-log wrapper (host side, before the step's forward, on the current stream) lands in the trace before that step's MoE calls, one per logged step, with gridY = step % 65535 + 1 (review 2). | G-STEPMARK on run A (`extract-gate-A`) and on the torch window (`extract-gate-B`). |
| A7 | The reset NVTX mark and the kernel timestamps share one nsys timeline. | G-REC: warmup markers before the mark are excluded, and the counts match. |
| A8 | `batch_descriptor.num_tokens` is the T the MoE layers see (padded rows are routed). | G-STEP (record n == logged T_pad); G-GRID uses the padded T. |
| A9 | Capture sizes are 1, 2, 4, 8..248 step 8, 256..512 step 16 at max_num_seqs 1024. | serve_capture's G-SIZES compares them with the engine's `cudagraph_capture_sizes`. |
| A10 | The torch profiler trace (Kineto) carries kernel `args.stream` and `args.grid`; with `async_scheduling=False` (mode B only) the window's records are exactly the window's kernels (every shard tagged profiler-window). | `extract-gate-B`, all gates. |
| A18 | vLLM 0.27.1 exposes its tokenizer loader (`vllm.tokenizers.get_tokenizer` or `vllm.transformers_utils.tokenizer.get_tokenizer`), which prompts.py uses so token ids match the engine's. | `prompts-build`. |
| A19 | With async scheduling on (modes A), the step log's per-step context (level, warmup flag) can lag one step at a level boundary; T, T_pad and the record join do not depend on it. | G-STEP on run A. |
| A11 | ERNIE's config.json keys (hidden_size 2560, moe_intermediate_size 1536, moe_num_experts 64, moe_k 6, 28 layers, MoE layers 1..27). | main.sh's gate60 runs G-CFG and the 27-layer bind count. Not checkable in smoke without un-blinding ERNIE, so it runs on the main rental before any long run. |
| A12 | Frontier's venv imports vLLM 0.27.1 and its trainer runs there. | The smoke: `frontier-setup` PASSED (functional_fused_experts branch); Frontier's profiler failed (no TP group). The trainer, hold-out and predict ran on Frontier db5a85b's code on the laptop (CPU, synthetic times); main: `frontier-train`. |
| A13 | Frontier's pinned stack (py3.10, torch 2.8 cu128 aarch64, vLLM 0.10.2 aarch64 wheel) installs and runs. | FALSE (smoke): no CUDA aarch64 torch 2.8. Dropped; Frontier is trained on our profiles. |
| A14 | The study's lock prelude holds 1710 and 1410 during serving. | Per-call SM clock from the NVML marks; off-lock calls are dropped and counted (`off_lock_dropped`). |
| A15 | The capture overhead (one small kernel per call plus a drain every ≤256 steps) does not distort serving. | The smoke's graph-mode versus node-mode runs (step intervals in `steps.jsonl`), and the marker share of step time in the sqlite. |
| A16 | `nvidia-smi -lgc` takes, and enforced.power.limit reads back 700 W (power.limit reads the 900 W default setting). | `env-preflight` P3/P4; `kit_lock`'s "GPU clocks set to" read-back; `power-readback`. |
| A17 | The estimator's T-histogram resembles the real one. | `expected_histogram.json` versus `manifest.json realized_per_tile` from the smoke run; the main run's G-COV is a hard gate. |

## Outputs

Everything goes under one results directory (`$RESULTS_ROOT/prod-{smoke,main}-<utc>`):

- `SUMMARY.txt`: one PASS/FAIL/SKIPPED line per step;
- `logs/<step>.log` (`offline-logs/{SUMMARY.txt,logs/}` for offline.sh);
- per run:
  - `ring/shard_*.npz`, `steps.jsonl`, `clocks.jsonl`;
  - `capture_manifest.json`, `manifest.json`, `gates.json`, `expected_histogram.json`, `sequences_by_path.json`;
  - `serve.nsys-rep`/`.sqlite` (main: under `$MOE_HOME/traces/<run>/<label>/`, outside the results tree, with
    `traces.sha256.json` in the tree);
  - `calls.jsonl`, `counts.npy` (or their shards with `calls.shards.json`), `sequences.json`;
- `sag/par.json` (the board's 128x128 par);
- `sample-*/`, `sample-probe-*/`, `replay-probe-*/` (the re-measure's probe and its `replay_wall.json`);
- `replay-*/private_weight_reference/<run id>/{cells.csv,report.json,cell_wall.json}` (smoke: `replay_wall.json`);
- `frontier-probe/{timer-<branch>/timer.json,points.json,profile/,models/,predictions/}` (the early probe);
- `frontier/timer-<branch>/timer.json` (main: the trace under `$MOE_HOME/traces/<run>/frontier/timer-<branch>/`),
  `frontier/points.json`, `frontier/profile{,-fills,-partial}/compute/gh200/<model>/moe_kernel_only.csv` with
  `csv_meta.json` (truth, coverage, missing points), `frontier/{models,models-simpath,models-fills,models-partial}
  (+ versions.json)`, `frontier/holdout/holdout.json`, `frontier/predictions/{index.json,*.npy}`;
- `replay-offset-<branch>/replay_check.json` (its trace under `$MOE_HOME/traces/<run>/replay-offset-<branch>/`;
  smoke: `replay-nsys-<branch>/` with the trace beside it);
- `score/prod.score.{json,txt}` (smoke on the VM; main after `offline.sh` on the laptop).
