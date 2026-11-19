"""One serving run of vLLM 0.27.1 with the routing capture on (KIT-SPEC 2). Offline engine, in this process.

    python scripts/prod/serve_capture.py --model ernie-4.5-21b-a3b --prompts P.jsonl --concurrency 1,8,24,48,80,112,256,1024x3 \\
        --requests-per-level 64 --mode graphs --shared-stream off --out DIR            # (under nsys: smoke.sh/main.sh)
    python scripts/prod/serve_capture.py ... --mode eager --torch-profiler 112:40 --out DIR
    python scripts/prod/serve_capture.py ... --dry-run          # the plan and the expected T histogram, no GPU

THE LOAD. Closed loop per concurrency level L: L requests are submitted at once (a prefill burst), then every
request that finishes is replaced by the next prompt, so L stay in flight until the level's quota (max(N, 2L))
is submitted; the level then drains. `LxR` runs level L R times (each with its own burst: the bursts are what
reach T >= 8256, the G16 tile). Prompts are taken in file order, each request generates exactly its max_tokens
(ignore_eos, temperature 0), seeds fixed. Chunked prefill on, max_num_batched_tokens 16384 (registered),
max_num_seqs 1024.

WHY THESE LEVELS (CRITIQUE M9: each default tile >= 30 distinct steps). Decode steps run T = the number of running
requests, so L in {1, 8, 24} lands on T <= 32, 48 on <= 64, 80 on <= 96, 112 on <= 128, 256 on <= 512 (plus mixed
prefill chunks); 1024 lands on > 512 and its bursts fill the 16384 budget (G16 at T >= 8256). `--dry-run` prints
the estimator's expected per-tile step counts (`estimate_histogram`, a CPU model of V1's decode-first chunked
prefill scheduler); the run's own step log is gated against the same minimum (G-COV) after the run.

RECORDED (manifest.json): vLLM version, every bound layer's MoE backend (must be TRITON: G-BACKEND stops the run
before any step), engine config (capture sizes, mode), env, the prompt set's header, GPU clocks/power at start
and end, plus a 10 Hz NVML sampler (clocks.jsonl) that also drops an NVTX mark per sample so nsys time maps to
the clock trace (per-call SM clock; off-lock calls dropped by the scorer).

EXIT: 0 every gate PASS; 3 a gate FAILED (written to gates.json, printed FAIL with the reason); 2 refused before
the run (backend not TRITON, a config mismatch, bad arguments).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402
import prompts as PR  # noqa: E402

MAX_BATCHED = 16384
MAX_SEQS = 1024
MIN_STEPS_PER_TILE = 30
WARMUP_STEPS = 4
DEFAULT_LEVELS = "1,8,24,48,80,112,256,1024x3"


# ------------------------------------------------------------------------------------------------------------
# levels and the T-histogram estimator (pure; tests/test_prod_prompts.py)
# ------------------------------------------------------------------------------------------------------------

def parse_levels(spec: str) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        L, _, R = tok.partition("x")
        L, R = int(L), int(R or 1)
        if not 1 <= L <= MAX_SEQS or R < 1:
            raise ValueError(f"level {tok!r}: 1 <= L <= {MAX_SEQS}, repeats >= 1")
        out += [L] * R
    if not out:
        raise ValueError("no concurrency levels")
    return out


def level_quota(L: int, n: int) -> int:
    return max(int(n), 2 * int(L))


@dataclass
class _Req:
    prefill_left: int
    decode_left: int


def simulate_level(L: int, reqs: list[tuple[int, int]], *, budget: int = MAX_BATCHED,
                   max_seqs: int = MAX_SEQS) -> list[int]:
    """Unpadded T per engine step for one closed-loop level: V1's scheduler model (running requests first in
    arrival order: a decoding request takes 1 token, a prefilling one min(left, budget); then waiting requests
    are admitted FCFS while running < min(L, max_seqs) and budget remains, chunked). A request whose prefill
    completes in a step emits its first token there; it then needs max_tokens - 1 decode steps."""
    pending = deque(reqs)
    waiting: deque = deque()
    running: list[_Req] = []
    cap = min(L, max_seqs)
    for _ in range(min(cap, len(pending))):
        waiting.append(pending.popleft())
    steps = []
    while running or waiting:
        b = budget
        T = 0
        for r in running:
            if b <= 0:
                break
            take = 1 if r.prefill_left == 0 else min(r.prefill_left, b)
            if r.prefill_left:
                r.prefill_left -= take
                if r.prefill_left == 0:
                    r.decode_left -= 1          # first token sampled at the end of the prefill
            else:
                r.decode_left -= 1
            T += take
            b -= take
        while waiting and len(running) < cap and b > 0:
            inp, out = waiting.popleft()
            take = min(inp, b)
            r = _Req(prefill_left=inp - take, decode_left=out)
            if r.prefill_left == 0:
                r.decode_left -= 1
            running.append(r)
            T += take
            b -= take
        if T == 0:
            raise RuntimeError("estimator stalled")
        steps.append(T)
        done = [r for r in running if r.prefill_left == 0 and r.decode_left <= 0]
        running = [r for r in running if not (r.prefill_left == 0 and r.decode_left <= 0)]
        for _ in done:
            if pending:
                waiting.append(pending.popleft())
    return steps


def estimate_histogram(levels: list[int], rows: list[dict], n_per_level: int, *, mode: str = "graphs",
                       sizes: list[int] | None = None) -> dict:
    """Expected steps per tile (padded T) over the run, prompts taken in order and wrapping."""
    i = 0
    per_tile: Counter = Counter()
    per_level = []
    steps_total = 0
    for L in levels:
        q = level_quota(L, n_per_level)
        reqs = []
        for _ in range(q):
            r = rows[i % len(rows)]
            reqs.append((int(r["input_len"]), int(r["max_tokens"])))
            i += 1
        Ts = simulate_level(L, reqs)
        tiles = Counter(PL.tile_bin(PL.padded_T(T, mode, sizes)) for T in Ts)
        per_tile.update(tiles)
        steps_total += len(Ts)
        per_level.append({"L": L, "requests": q, "steps": len(Ts), "tiles": dict(tiles)})
    return {"per_tile": {lab: int(per_tile.get(lab, 0)) for lab, _, _ in PL.TILE_BINS}, "per_level": per_level,
            "steps": steps_total}


def coverage_gate(per_tile: dict, minimum: int) -> dict:
    short = {k: v for k, v in per_tile.items() if v < minimum}
    return {"verdict": "PASS" if not short else "FAIL", "minimum": minimum, "per_tile": per_tile,
            "why": "" if not short else f"tiles under {minimum} steps: {short}"}


# ------------------------------------------------------------------------------------------------------------
# GPU side
# ------------------------------------------------------------------------------------------------------------

def nvml_snapshot() -> dict:
    try:
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        out = {"name": N.nvmlDeviceGetName(h), "sm_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
               "mem_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM),
               "power_w": N.nvmlDeviceGetPowerUsage(h) / 1000.0,
               "power_limit_w": N.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
               "temp_c": N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
               "driver": N.nvmlSystemGetDriverVersion()}
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in out.items()}
    except Exception as exc:                                         # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


class ClockSampler(threading.Thread):
    """10 Hz NVML sampler -> clocks.jsonl; one NVTX mark 'clk <i>' per sample (maps nsys time to samples)."""

    def __init__(self, path: Path, period_s: float = 0.1):
        super().__init__(daemon=True)
        self.path, self.period, self.stop_ev = path, period_s, threading.Event()

    def run(self):
        try:
            import pynvml as N
            import torch
            N.nvmlInit()
            h = N.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:                                     # noqa: BLE001
            self.path.write_text(json.dumps({"error": str(exc)}) + "\n")
            return
        i = 0
        with open(self.path, "a") as fh:
            while not self.stop_ev.is_set():
                t = time.monotonic_ns()
                try:
                    row = {"i": i, "t_ns": t, "sm_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                           "power_w": N.nvmlDeviceGetPowerUsage(h) / 1000.0}
                    torch.cuda.nvtx.mark(f"clk {i}")
                except Exception as exc:                             # noqa: BLE001
                    row = {"i": i, "t_ns": t, "error": str(exc)}
                fh.write(json.dumps(row) + "\n")
                i += 1
                self.stop_ev.wait(self.period)


def resolve_revision(repo: str, rev: str | None) -> str | None:
    """The commit sha of the snapshot the engine loaded, read from the local HF cache only (no network): the
    snapshot directory's name."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, revision=rev, local_files_only=True)).name
    except Exception:                                                # noqa: BLE001
        return None


def write_gates(out: Path, gates: dict) -> int:
    (out / "gates.json").write_text(json.dumps(gates, indent=2, default=str))
    bad = [k for k, v in gates.items() if v.get("verdict") == "FAIL"]
    for k, v in gates.items():
        print(f"{v.get('verdict', '?'):5s} {k}" + (f": {v.get('why')}" if v.get("why") else ""))
    return 3 if bad else 0


def run(a, model: PL.ServedModel, ps: dict, levels: list[int]) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1" if a.shared_stream == "off" else "0"
    import random

    import numpy as np
    import torch
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    import moe_capture as MC
    import vllm
    from vllm import LLM, SamplingParams

    manifest = {"argv": sys.argv, "model": model.key, "hf_repo": model.hf_repo, "vllm": vllm.__version__,
                "torch": torch.__version__, "python": platform.python_version(), "mode": a.mode,
                "shared_stream": a.shared_stream, "levels": levels, "requests_per_level": a.requests_per_level,
                "max_num_batched_tokens": a.max_num_batched_tokens, "max_num_seqs": MAX_SEQS,
                "prompts": ps["header"], "seed": a.seed,
                "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "CUDA_", "TORCH", "NCCL_"))},
                "gpu_start": nvml_snapshot()}
    gates: dict = {}
    if vllm.__version__ != "0.27.1":
        gates["G-VLLM"] = {"verdict": "FAIL", "why": f"vLLM {vllm.__version__}, the kit is for 0.27.1"}
        return write_gates(out, gates)
    rev = a.revision or PL.hf_revision(model.key)
    if rev is None and a.require_pinned:
        gates["G-REV"] = {"verdict": "FAIL", "why": f"{model.key} has no pinned HF revision (prodlib.HF_REVISIONS)"}
        return write_gates(out, gates)
    MC.install(out, top_k=model.cfg.top_k, n_moe_layers=model.moe_layers, warm_sizes=PL.capture_sizes())
    kw = dict(model=model.hf_repo, dtype="bfloat16", seed=a.seed, max_num_batched_tokens=a.max_num_batched_tokens,
              max_num_seqs=MAX_SEQS, enable_chunked_prefill=True, enforce_eager=(a.mode == "eager"),
              gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
              trust_remote_code=True, enable_prefix_caching=False)
    if a.torch_profiler:
        # mode B: synchronous scheduling, so no step of the next batch is in flight when the window opens or
        # closes (the window's records must be exactly the window's kernels; G-REC checks it)
        kw["async_scheduling"] = False
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str((out / "torch_trace").resolve()),
                                 "torch_profiler_with_stack": False, "torch_profiler_use_gzip": False,
                                 "torch_profiler_record_shapes": False}
    if rev:
        kw.update(revision=rev, tokenizer_revision=rev)
    llm = LLM(**kw)
    cap = MC.current()
    resolved = resolve_revision(model.hf_repo, rev)
    manifest["hf_revision"] = {"pinned": rev, "resolved": resolved}
    if rev and resolved and resolved != rev:
        gates["G-REV"] = {"verdict": "FAIL", "why": f"loaded snapshot {resolved}, pinned {rev}"}
    else:
        gates["G-REV"] = {"verdict": "PASS", "why": "" if rev else f"UNPINNED: resolved {resolved} (pin it in "
                                                                   "prodlib.HF_REVISIONS before the kit freezes)"}
    hf = llm.llm_engine.model_config.hf_config.to_dict()
    bad_cfg = PL.check_hf_config(model, hf)
    backends = sorted({r["backend"] for r in cap.bound_layers})
    gates["G-CFG"] = {"verdict": "FAIL" if bad_cfg else "PASS", "why": "; ".join(bad_cfg)}
    gates["G-BACKEND"] = {"verdict": "PASS" if backends == ["TRITON"] else "FAIL", "backends": backends,
                          "experts_cls": sorted({r["experts_cls"] for r in cap.bound_layers}),
                          "why": "" if backends == ["TRITON"] else f"bound layers report {backends}, not TRITON"}
    cc = llm.llm_engine.vllm_config.compilation_config
    manifest["engine"] = {"cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", None) or []),
                          "cudagraph_mode": str(getattr(cc, "cudagraph_mode", "")),
                          "compilation_mode": str(getattr(cc, "mode", ""))}
    if a.mode == "graphs" and sorted(manifest["engine"]["cudagraph_capture_sizes"]) != PL.capture_sizes():
        gates["G-SIZES"] = {"verdict": "FAIL", "why": f"capture sizes {manifest['engine']['cudagraph_capture_sizes']}"
                                                      f" differ from the kit's {PL.capture_sizes()}"}
    if any(v["verdict"] == "FAIL" for v in gates.values()):
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        cap.steps.close()
        write_gates(out, gates)
        return 2
    sampler = ClockSampler(out / "clocks.jsonl")
    sampler.start()
    cap.reset()
    eng = llm.llm_engine
    rows = ps["rows"]
    nxt = 0
    rid = 0
    prof_level, prof_steps = (None, 0)
    if a.torch_profiler:
        lv, _, ns = a.torch_profiler.partition(":")
        prof_level, prof_steps = int(lv), int(ns or 40)
    profiled = False
    t0 = time.time()
    for li, L in enumerate(levels):
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            print(f"--max-seconds {a.max_seconds} reached: levels from {li} on not run", flush=True)
            manifest["stopped_at_level"] = li
            break
        quota = level_quota(L, a.requests_per_level)
        submitted = inflight = 0
        step_in_level = 0

        def submit():
            nonlocal nxt, rid, submitted, inflight
            r = rows[nxt % len(rows)]
            nxt += 1
            sp = SamplingParams(max_tokens=int(r["max_tokens"]), min_tokens=int(r["max_tokens"]), ignore_eos=True,
                                temperature=0.0, seed=a.seed)
            eng.add_request(f"r{rid}", {"prompt_token_ids": r["prompt_token_ids"]}, sp)
            rid += 1
            submitted += 1
            inflight += 1
        for _ in range(min(L, quota)):
            submit()
        prof_on = False
        while eng.has_unfinished_requests():
            cap.context = {"level_idx": li, "L": L, "warmup": step_in_level < WARMUP_STEPS, "profiled": prof_on}
            if (a.torch_profiler and not profiled and not prof_on and L == prof_level
                    and step_in_level == WARMUP_STEPS):
                cap.drain(tag="pre-profile")
                cap.drain_tag = "profiler-window"
                llm.start_profile()
                prof_on, prof_left = True, prof_steps
                cap.context["profiled"] = True
            outs = eng.step()
            step_in_level += 1
            if prof_on:
                prof_left -= 1
                if prof_left <= 0:
                    torch.cuda.synchronize()
                    llm.stop_profile()
                    cap.drain(tag="profiler-window")
                    cap.drain_tag = "periodic"
                    prof_on, profiled = False, True
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                quota = submitted                      # stop feeding; the level drains
            for o in outs:
                if getattr(o, "finished", False):
                    inflight -= 1
                    if submitted < quota:
                        submit()
        print(f"level {li} L={L}: {submitted} requests, {step_in_level} steps, {time.time() - t0:.0f} s", flush=True)
    if a.torch_profiler and not profiled:
        gates["G-PROFILER"] = {"verdict": "FAIL", "why": f"the profiler window at L={prof_level} never ran"}
    sampler.stop_ev.set()
    sampler.join(timeout=2)
    cap.close()
    manifest["gpu_end"] = nvml_snapshot()
    manifest["seconds"] = round(time.time() - t0, 1)
    steps = PL.read_jsonl(out / "steps.jsonl")
    real = Counter(PL.tile_bin(s["T_pad"]) for s in steps if not s.get("dummy") and not s.get("warmup"))
    manifest["realized_per_tile"] = {lab: int(real.get(lab, 0)) for lab, _, _ in PL.TILE_BINS}
    gates["G-COV"] = coverage_gate(manifest["realized_per_tile"], a.min_steps_per_tile)
    gates["G-DUMMY"] = {"verdict": "PASS" if not any(s.get("dummy") for s in steps) else "FAIL",
                        "why": "" if not any(s.get("dummy") for s in steps) else "dummy runs after the reset"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return write_gates(out, gates)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--concurrency", default=DEFAULT_LEVELS)
    ap.add_argument("--requests-per-level", type=int, default=64)
    ap.add_argument("--max-tokens-dist", default=None,
                    help="override the prompt file's output lengths: fixed:N | uniform:A,B | lognormal:MU,SIGMA")
    ap.add_argument("--mode", choices=("graphs", "eager"), default="graphs")
    ap.add_argument("--max-num-batched-tokens", type=int, default=MAX_BATCHED)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--shared-stream", choices=("on", "off"), default="off")
    ap.add_argument("--torch-profiler", default=None, metavar="L:STEPS",
                    help="eager only: vLLM's torch profiler around STEPS engine steps of level L")
    ap.add_argument("--min-steps-per-tile", type=int, default=MIN_STEPS_PER_TILE)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop feeding requests after this long (the 60 s gate runs); 0: no limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None,
                    help="the model's HF commit (default prodlib.HF_REVISIONS); pins the model and the tokenizer")
    ap.add_argument("--require-pinned", action="store_true",
                    help="refuse a model with no pinned revision (main.sh); the smoke runs unpinned and prints the sha")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        levels = parse_levels(a.concurrency)
        if a.max_num_batched_tokens != MAX_BATCHED:
            raise ValueError(f"--max-num-batched-tokens is registered at {MAX_BATCHED} (CRITIQUE M5)")
        if a.torch_profiler and a.mode != "eager":
            raise ValueError("--torch-profiler is mode B (eager) only")
        ps = PR.load(a.prompts)
        if a.max_tokens_dist:
            import random
            dist = PR.parse_out_dist(a.max_tokens_dist)
            rng = random.Random(a.seed)
            for r in ps["rows"]:
                r["max_tokens"] = PR.draw_out_len(dist, rng, None, ps["header"]["min_out"], ps["header"]["max_out"])
            ps["header"] = dict(ps["header"], out_dist_override=a.max_tokens_dist)
    except (ValueError, PR.PromptError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    model = PL.MODELS[a.model]
    est = estimate_histogram(levels, ps["rows"], a.requests_per_level, mode=a.mode)
    print(f"serve_capture: {model.key} ({model.hf_repo}) mode={a.mode} shared-stream={a.shared_stream} "
          f"levels={levels} N/level={a.requests_per_level} out={a.out}")
    print(f"expected steps {est['steps']}; per tile (padded T):")
    for k, v in est["per_tile"].items():
        flag = "" if v >= a.min_steps_per_tile else f"   < {a.min_steps_per_tile}"
        print(f"  {k:<22s}{v:8d}{flag}")
    if a.dry_run:
        print("DRY RUN: nothing launched")
        return 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "expected_histogram.json").write_text(json.dumps(est, indent=2))
    return run(a, model, ps, levels)


if __name__ == "__main__":
    sys.exit(main())
