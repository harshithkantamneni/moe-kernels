# Production-check kit (scripts/prod)

The kit measures vLLM 0.27.1's fused MoE call inside a real serving run on a GH200 and scores the frozen model
(v8), a roofline and Frontier against it. The numbers are descriptive: no bars, no verdicts. The spec is
KIT-SPEC.md, with PLAN.md, CAPTURE.md and CRITIQUE-DISPOSITION.md (2026-11-19). Nothing pinned or frozen is
edited. moe/spec.py is not touched: the ERNIE MoEConfig is built in `prodlib.py`.

## Files

| file | role |
|---|---|
| `moe_capture.py` | The hook inside the vLLM process. One Triton kernel per routed call writes topk_ids (int16) and a record (layer, padded n, offset, seq) into a device ring. The host drains the ring between steps into npz shards. A step log records (step, T, padded T, cudagraph mode). |
| `serve_capture.py` | One offline serving run: closed-loop concurrency levels, graphs or eager, shared-experts stream on/off, optional torch-profiler window. Writes the manifest, the gates and a 10 Hz NVML clock log. Also holds the T-histogram estimator. |
| `prompts.py` | The deterministic prompt set: ShareGPT, with a Gutenberg text as fallback. Each source is pinned by URL and sha256. The set file is tamper-evident (it carries the sha256 of its own body). |
| `nsys_extract.py` | Reads the nsys sqlite export (mode A) or the torch-profiler trace (mode B). Splits the timeline into calls (align..moe_sum, router excluded) and reports kernel-sum and span per call. Runs the gates G-REC, G-GEMM, G-STEP, G-IDS, G-GRID and G-CLASS. |
| `replay.py` | Picks the registered stratified sample. Re-times each sampled call with r3_decode's GRR instrument on its real ids. Can run under locked_r3. |
| `frontier_setup.sh`, `frontier_op.py` | Frontier at db5a85b in its own venvs: primary (vLLM 0.27.1) and fidelity (its pinned stack). Profile, train, predict and hold-out, run as Frontier's docs run them. |
| `score_prod.py` | Joins the calls with the v8 prices, the roofline and Frontier. Reports MAPE, wMAPE, p50/p95 \|e\| and bias, each with step-bootstrap CIs, per tile, branch and stratum, plus per-step totals and the run deltas. |
| `smoke.sh`, `main.sh` | The VM sequences, fail-fast. Each step prints PASS/FAIL with its reason. `--dry-run` prints the steps and runs nothing. |
| `prodlib.py` *(extra)* | Code several scripts share: the models, tiles, branches and strata, the capture sizes, the ring format and decoder, and **`price_v8`, the one v8 binding** (`PRICER_MODULE`, or `--pricer` / `$PROD_PRICER`). |
| `kitlib.sh` *(extra)* | The step runner, the lock and unlock helpers, nsys location, and the env and power checks that both sequences use. |
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
     records the variant count.
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

   Concurrency design: the default levels are `1,8,24,48,80,112,256,1024x3`, with N 64 per level and quota
   max(N, 2L). The estimator (a model of V1's decode-first chunked-prefill scheduler) predicts, on a
   ShareGPT-like length mix, these steps per tile:

   | tile | steps |
   |---|---|
   | T≤32 | 21.7k |
   | ≤64 | 1051 |
   | ≤96 | 764 |
   | ≤128 | 565 |
   | ≤512 | 1722 |
   | >512 | 1415 |
   | G16 (≥8256) | 55, all from the three 1024 bursts |

   Every tile is ≥30, which a test asserts. After the run, G-COV checks the realized histogram against the same
   minimum.
3. **prompts.py.** Covered by the prompt tests: a deterministic build, the pin check, and refusal of an edited
   set.
4. **nsys_extract.py.** Primary alignment: the k-th marker kernel after the reset ↔ the k-th ring record. NVTX
   is recorded only as a secondary signal. Classification and the gates are described in its docstring. Truth is
   measured two ways: kernel-sum (classified kernels only; an intruder such as a shared-expert GEMM is counted
   and excluded) and span. Per-call SM clock comes from the NVML sampler's `clk i` NVTX marks.
5. **replay.py.** It reuses r3_decode's `GPUBackend` (GRR rotating weight copies, a CUDA graph of back-to-back
   calls, `launch_floor.mirror_time_cell`) and its `run_page`, spy and V-ALIGN, GD4/GD5, COPIES, GD1,
   `Store` and `write_report`, all unchanged. **The adapter** (smallest possible, disclosed) does two things:
   - it builds the `DecodePage` straight from the sample, which bypasses `page_from`'s T≤128 check;
   - its `RealIdsMixin.build` passes the call's logged ids instead of `realize_counts`.

   Each branch is timed at its own clock (decode 1710, prefill 1410) through `locked_r3.py --r3-script
   scripts/prod/replay.py`. The plan's `session`/`WRITES TO` lines are r3_decode's, so locked_r3 parses them,
   and a test checks this. The sample rule is fixed: an equal quota per tile, systematic within a tile, seed
   20261119; main.sh splits 1500 as 857 decode and 643 prefill. One extra replay runs under nsys node tracing to
   measure the instrument offset.
6. **Frontier.** It is driven as its docs drive it: `examples/profiling/profile_moe.sh` (record_function = KERNEL_ONLY,
   `--max-tokens 16384`, `-- --enable_load_imbalance --load_distributions …`), `python -m
   frontier.training.cli moe`, and the trained `moe_grouped_gemm` estimator, with features from Frontier's own
   `MoELoadImbalanceInput(expert_token_counts=real counts)`. The API branch Frontier picks is read off its own
   print line and recorded in `api.json`. The fidelity arm is scored only on its own held-out points.
7. **score_prod.py.** Descriptive only. Pricing runs in a process pool, cached by (T, counts digest, ids
   digest). A locked run scores only its own branch. A default-clock run is labelled a production delta only.
   Off-lock calls (more than 15 MHz from the lock) are dropped and counted. The refused-call share of MoE time
   is reported.
8–9. **smoke.sh and main.sh.** They follow CRITIQUE-DISPOSITION RENTALS (1) and (2). main.sh runs the ERNIE 60 s
   gate before any long run; a test asserts the order.

## Disclosures (Frontier and the replay)

- Frontier's code is not edited: frontier_setup.sh checks `git diff --quiet` before and after setup. One
  **untracked data file** is added, `data/config/models/<name>.json`. It describes the served model's routed MoE
  operator in the HF keys Frontier's loader reads. It sets `model_type qwen3_moe` so Frontier's architecture
  registry resolves. Its attention fields are not used by the MoE profiler.
- **The simpath shim** (`frontier_op.py _simpath`). Frontier's MoETrainer is used with its feature selection
  swapped for the simulator's `select_moe_operator_features`: (total_routed_tokens, num_tokens_post_padded) when
  the table records the padded count, which only the 0.10.x low-level branch does. At prediction time BM comes
  from Frontier's own `block_size_m_for_tokens` over the table's ranges (CRITIQUE M1). The documented CLI model
  is reported separately.
- The primary venv is the study's vLLM closure (`requirements/resolved-vllm.txt`, with the editable line
  dropped as setup_runpod.sh does) plus Frontier's dependencies, constrained by that closure. Frontier is
  installed `-e --no-deps`.
- Replay differences from serving: random hidden states, and private cold weight copies (serving's weights may
  be warm in L2 at small T). The ids are the real ones, including padded rows.
- The capture marker kernel runs between the router and align. It is outside both truths, but it does add a
  small cost to every serving step.

## Before the kit freezes (after the smoke rental)

- Paste the ShareGPT sha256 that smoke prints into `prompts.py SOURCES["sharegpt"]["sha256"]`. Until then main.sh
  stops at `prompts-fetch`.
- v8 lands: `price_v8` imports `r3_timing_model_v8.price` (switch `PL.PRICER_MODULE` if the name differs).
  main.sh's `pricer-import` step fails if it is missing.
- **Roofline constants:** v8 should expose `ROOFLINE = {"decode": {"T0_ms", "bw_gbps"}, "prefill": {…}}` at its
  frozen constants, or a registered `--roofline` JSON must be passed. Without either, the roofline is reported as
  NOT SCORED.
- **Launcher:** the `vm_run*.sh` `--driver` allow-lists do not name `scripts/prod/*.sh`. Either a `vm_run_prod.sh`
  copy is needed (not built here, because the existing launchers are not edited) or the sequences are started
  over ssh after `setup_vm.sh`. Lambda launch and terminate stay with `~/.lambda/status` / `~/.lambda/rent`.
- `PROD_POWER_W` (default 700) is the power limit that main.sh's power-readback requires.

## Assumptions that cannot be verified without the GPU (each is checked in the smoke rental)

| # | assumption | how smoke.sh verifies it |
|---|---|---|
| A1 | The router call runs inside the opaque MoE custom op, so a capture fn bound after torch.compile's profile run still runs, in both eager and graph replay. | `extract-gate-A`: G-REC (ring records == marker kernels) and G-STEP (n_layers records per logged step, n == T_pad). |
| A2 | Triton's scalar `tl.atomic_add` executes once per program and returns the old value; `if ok:` on a scalar compiles. | The same G-REC/G-STEP, plus `decode_shards`' seq/offset checks on every shard. |
| A3 | `do_not_specialize` yields one compiled variant for every n, so no JIT compile falls inside a capture. | `capture_manifest.json kernel_variants` (expect 1 per ids dtype). A capture-time compile would crash the run. |
| A4 | The unquantized method is modular (`is_monolithic` False) and the backend on SM90 is TRITON (`TritonExperts`). | `serve_capture` G-BACKEND; `bind` refuses a monolithic method. |
| A5 | The per-call kernel sequence is align (small-batch / align+sort ± memset / naive fill), fused_moe_kernel, act_and_mul, fused_moe_kernel, moe_sum, with no chunking (envs.py v0.27.1 has no VLLM_FUSED_MOE_CHUNK_SIZE). | G-CLASS, G-GEMM (2 × records); `sequences.json` lists the 12 most common name sequences, for a human check. |
| A6 | nsys `--cuda-graph-trace=node` works on aarch64 GH200 with driver 580 and reports graph-node kernels with their gridX on the launch stream. | G-REC/G-GRID on run A; `gates.json marker_streams`. If it fails, mode B (eager, torch profiler) is the fallback. |
| A7 | The reset NVTX mark and the kernel timestamps share one nsys timeline. | G-REC: warmup markers before the mark are excluded, and the counts match. |
| A8 | `batch_descriptor.num_tokens` is the T the MoE layers see (padded rows are routed). | G-STEP (record n == logged T_pad); G-GRID uses the padded T. |
| A9 | Capture sizes are 1, 2, 4, 8..248 step 8, 256..512 step 16 at max_num_seqs 1024. | serve_capture's G-SIZES compares them with the engine's `cudagraph_capture_sizes`. |
| A10 | The torch profiler trace (Kineto) carries kernel `args.stream` and `args.grid`; with `async_scheduling=False` (mode B only) the window's records are exactly the window's kernels (every shard tagged profiler-window). | `extract-gate-B`, all gates. |
| A18 | vLLM 0.27.1 exposes its tokenizer loader (`vllm.tokenizers.get_tokenizer` or `vllm.transformers_utils.tokenizer.get_tokenizer`), which prompts.py uses so token ids match the engine's. | `prompts-build`. |
| A19 | With async scheduling on (modes A), the step log's per-step context (level, warmup flag) can lag one step at a level boundary; T, T_pad and the record join do not depend on it. | G-STEP on run A. |
| A11 | ERNIE's config.json keys (hidden_size 2560, moe_intermediate_size 1536, moe_num_experts 64, moe_k 6, 28 layers, MoE layers 1..27). | main.sh's gate60 runs G-CFG and the 27-layer bind count. Not checkable in smoke without un-blinding ERNIE, so it runs on the main rental before any long run. |
| A12 | Frontier's primary venv imports vLLM 0.27.1 and takes the `functional_fused_experts` branch; Frontier's times are in ms. | `frontier-setup` (version and branch print) and `api.json`. The hold-out MAPE and the predict magnitudes are compared with the replay's ms in the smoke score. |
| A13 | Frontier's pinned stack (py3.10, torch 2.8 cu128 aarch64, vLLM 0.10.2 aarch64 wheel) installs and runs. | `frontier-setup` / `frontier-profile-fidelity` (soft: reported and dropped if not). |
| A14 | The study's lock prelude holds 1710 and 1410 during serving. | Per-call SM clock from the NVML marks; off-lock calls are dropped and counted (`off_lock_dropped`). |
| A15 | The capture overhead (one small kernel per call plus a drain every ≤256 steps) does not distort serving. | The smoke's graph-mode versus node-mode runs (step intervals in `steps.jsonl`), and the marker share of step time in the sqlite. |
| A16 | `nvidia-smi -lgc` takes, and power.limit reads back 700 W. | `kit_lock`'s "GPU clocks set to" read-back; `power-readback`. |
| A17 | The estimator's T-histogram resembles the real one. | `expected_histogram.json` versus `manifest.json realized_per_tile` from the smoke run; the main run's G-COV is a hard gate. |

## Outputs

Everything goes under one results directory (`$RESULTS_ROOT/prod-{smoke,main}-<utc>`):

- `SUMMARY.txt`: one PASS/FAIL line per step;
- `logs/<step>.log`;
- per run:
  - `ring/shard_*.npz`, `steps.jsonl`, `clocks.jsonl`;
  - `capture_manifest.json`, `manifest.json`, `gates.json`, `expected_histogram.json`;
  - `serve.nsys-rep`/`.sqlite`;
  - `calls.jsonl`, `counts.npy`, `sequences.json`;
- `sample-*/`;
- `replay-*/private_weight_reference/<run id>/{cells.csv,report.json}`;
- `frontier/<arm>/{profile,models,holdout}`, `frontier/predictions.jsonl`;
- `score/prod.score.{json,txt}`.
