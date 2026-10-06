#!/usr/bin/env python
"""THE INSTRUMENTED KERNEL ON THE CARD: the perturbation gate and the stamp pages (rental 4,
owner decisions 2 and 5 of 2026-10-06).

    python scripts/instr_probe.py --mode perturb --variants FILE --out DIR [--dry-run]
    python scripts/instr_probe.py --mode stamps --variants FILE --variant ID --gate FILE --out DIR

R3'S OWN CALL. Weights, inputs, routing and the call come from private_weight_reference
(`build_private_weights`, `arm_inputs`, `arm_call`, `declared_experts`, `pinned_config`,
`counter_declaration`, `SWEEP.find_override`), imported and never copied, so a stamped
cell is a cell of the published pages. The kernel is swapped by `moe.instrumented.install`
(vLLM's module attribute, restored on exit); nothing else in the call changes.

A VARIANT is one instrumented unit of the plan: model, config (BLOCK_K, num_stages), its
cells (G x arm x n) and its instrument spec. The driver writes the plan's variants to FILE.

PERTURB (decision 2). For every variant, on its own cells (an eviction-hint byte unit: its G
at treads 1, 5, 9, every arm), the PLAIN kernel (`timed_plain`) and the COPY with the
variant's stamps and NO hint (`install`; a byte unit's copy is all-off) are timed the same
way: a CUDA event pair around each fused_moe_kernel launch, an L2 flush before every call
OUTSIDE the event pairs, the GPU held ahead of the host by a `torch.cuda._sleep` so no
launch waits on Python, CALLS calls a burst, REPS bursts each in alternating order. Per
(cell, GEMM): r = median plain ms / median copy ms. The gate, per variant (gate_verdict):
median |r - 1| <= 1%, worst <= 2%; plain and copy paired BY CONFIG with equal registers,
shared and CTAs per SM (read off each launch's CompiledKernel); the ALL-OFF copy's SASS
equal to the plain kernel's with line info stripped (cuobjdump; unread FAILS); a hinted
unit's compiled PTX carrying L2::cache_hint (Triton 3.7.1 drops the hint on pipelined
cp.async loads, so a hinted unit FAILS here unless the hint survives); and the installed
vLLM fused_moe.py being the upstream file. Writes DIR/perturb.json, DIR/gate.env
(`GATE_<id>=PASS|FAIL`) and the SASS it compared under DIR/sass/.

STAMPS. One variant whose gate line reads PASS (else REFUSED, exit 2, nothing launched):
per cell WARMUP calls, then MEASURED calls with the stamps on, each launch's int64 rows
saved as DIR/stamps/G<g>-<arm>-n<n>-call<c>-<w1|w2>.npy and indexed in DIR/stamps.json
(grid, N, K, EM, the config, the event-pair ms of the launch), an L2 flush before every
measured call, and each compiled config's SASS under DIR/sass/ with its bracket checks
(moe.instrumented.cubin.sass_checks), which the scorer requires.

EXIT CODES: 0 done (every gate PASS for perturb); 1 perturb ran and some gate FAILED; 2
refused (bad variants, no CUDA / vLLM, the gate not PASS; and --dry-run); 3 a cell failed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import private_weight_reference as PW  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

SWEEP = PW.SWEEP
BLOCK_M = 32
BLOCK_N = 64
#: owner decision 2 (2026-10-06); docs/registered/2026-10-06-rental4-perturb-gh200.json
#: carries the same numbers (a test holds them equal)
PERTURB_TOL = {"median_max": 0.01, "worst_max": 0.02, "occupancy": "identical"}
REPS = 8
CALLS = 10
STAMP_WARMUP = 3
STAMP_CALLS = 2
#: an eviction-hint byte unit's gate cells: its G at these treads, every arm
BYTE_GATE_TREADS = (1, 5, 9)
#: the sleep that holds the GPU ahead of a burst: cycles = ms x the lock in kHz
SLEEP_AHEAD_MS = 25.0
KINDS = ("stamps", "bytes")


class Refused(Exception):
    pass


# --------------------------------------------------------------------------
# variants and their cells (pure)
# --------------------------------------------------------------------------

def check_variant(v: dict) -> dict:
    """One variant, validated as R3 would (ladder, config, arms), with its spec parsed and
    its gate spec and gate cells resolved."""
    need = ("id", "kind", "model", "groups", "treads", "arms", "spec")
    miss = [k for k in need if k not in v]
    if miss:
        raise Refused(f"variant {v.get('id')!r} has no {miss}")
    if v["kind"] not in KINDS:
        raise Refused(f"variant {v['id']}: kind {v['kind']!r} is not one of {KINDS}")
    if v["model"] not in MODEL_CONFIGS:
        raise Refused(f"variant {v['id']}: unknown model {v['model']!r}")
    cfg = MODEL_CONFIGS[v["model"]]
    try:
        spec = I.parse_spec(v["spec"])
    except I.SpecError as exc:
        raise Refused(f"variant {v['id']}: {exc}") from None
    if v["kind"] == "stamps" and (not spec.stamps or spec.hints):
        raise Refused(f"variant {v['id']}: a stamps unit carries stamps and no hint")
    if v["kind"] == "bytes" and (spec.stamps or not spec.hints):
        raise Refused(f"variant {v['id']}: a byte unit carries hints and no stamps")
    arms = [str(a) for a in v["arms"]]
    if not arms or set(arms) - set(PW.ARMS):
        raise Refused(f"variant {v['id']}: arms {arms} are not a subset of {list(PW.ARMS)}")
    bk = int(v.get("block_k") or SWEEP.FIXED["BLOCK_SIZE_K"])
    st = int(v.get("num_stages") or SWEEP.FIXED["num_stages"])
    why = PW.config_refusal(cfg, block_m=BLOCK_M, block_n=BLOCK_N, block_k=bk, num_stages=st)
    if why:
        raise Refused(f"variant {v['id']}: the config cannot run: {why}")
    ladder = (PW.ladder_treads(cfg, BLOCK_M, PW.NATIVE_COUNTER_MAX_TREADS) if PW.native_only(arms)
              else PW.counter_ladder(cfg, BLOCK_M))
    treads = [int(n) for n in v["treads"]]
    off = [n for n in treads if n not in ladder]
    if off or not treads:
        raise Refused(f"variant {v['id']}: treads {off or treads} are not on R3's ladder {ladder}")
    groups = [int(g) for g in v["groups"]]
    if not groups or min(groups) < 1:
        raise Refused(f"variant {v['id']}: groups {groups}")
    if v["kind"] == "bytes":
        gate_spec = I.InstrSpec()
        gate_cells = [(g, a, n) for g in groups for a in PW.ARMS for n in BYTE_GATE_TREADS if n in ladder]
    else:
        gate_spec = I.InstrSpec(stamps=spec.stamps, every=spec.every, marks=spec.marks)
        gate_cells = [(g, a, n) for g in groups for a in arms for n in treads]
    return {**v, "cfg": cfg, "spec_obj": spec, "gate_spec": gate_spec, "block_k": bk,
            "num_stages": st, "groups": groups, "treads": treads, "arms": arms,
            "cells": [(g, a, n) for g in groups for a in arms for n in treads],
            "gate_cells": gate_cells}


def load_variants(path: Path) -> list[dict]:
    vs = json.loads(Path(path).read_text())
    if not isinstance(vs, list) or not vs:
        raise Refused(f"{path}: a JSON list of variants")
    ids = [v.get("id") for v in vs]
    if len(set(ids)) != len(ids):
        raise Refused(f"{path}: duplicate variant ids {ids}")
    return [check_variant(v) for v in vs]


def compare_configs(plain: dict, copy: dict) -> list[dict]:
    """Pair the plain and copy kernels BY CONFIG (moe.instrumented.config_key: tile, G, stages,
    warps, MUL_ROUTED_WEIGHT, top_k) and compare registers, shared and CTAs per SM; every
    plain config needs a copy of the same config. Values are compiled_facts dicts."""
    out = []
    for key in sorted(set(plain) | set(copy), key=str):
        p, c = plain.get(key), copy.get(key)
        row = {"config": list(key), "plain": None if p is None else {k: p[k] for k in ("regs", "shared")},
               "copy": None if c is None else {k: c[k] for k in ("regs", "shared")}}
        if p is None or c is None or p.get("occupancy") is None or c.get("occupancy") is None:
            row["equal"] = None
        else:
            row["plain"]["ctas_per_sm"] = p["occupancy"]["ctas_per_sm"]
            row["copy"]["ctas_per_sm"] = c["occupancy"]["ctas_per_sm"]
            row["equal"] = row["plain"] == row["copy"]
        out.append(row)
    return out


def gate_verdict(ratios: list, configs: list[dict], *, upstream_ok: bool = True,
                 sass_equal: bool | None = None, hint_ok: bool | None = None,
                 tol: dict = PERTURB_TOL) -> dict:
    """Decision 2 on one variant with the review's legs (build-r4-review sections 1 and 2):
    timing: median |r - 1| <= 1% and worst <= 2% over its (cell, GEMM) ratios;
    occupancy: every config paired, with equal registers, shared and CTAs per SM;
    SASS: the ALL-OFF copy's SASS equals the plain kernel's, line info stripped, on every
      config (None: unread, which FAILS);
    hint: a hinted unit's compiled PTX carries L2::cache_hint (None: no hint asked);
    upstream: the installed vLLM fused_moe.py is the upstream file."""
    why = []
    if not upstream_ok:
        why.append("the installed vLLM fused_moe.py is not the upstream file the copy came from")
    dev = [abs(r - 1) for r in ratios if r is not None]
    if not dev or len(dev) < len(ratios):
        why.append(f"{len(ratios) - len(dev)} of {len(ratios)} ratios unread")
    med = statistics.median(dev) if dev else None
    worst = max(dev) if dev else None
    if med is not None and med > tol["median_max"]:
        why.append(f"median |r - 1| {100 * med:.2f}% > {100 * tol['median_max']:.0f}%")
    if worst is not None and worst > tol["worst_max"]:
        why.append(f"worst |r - 1| {100 * worst:.2f}% > {100 * tol['worst_max']:.0f}%")
    if not configs:
        why.append("occupancy unread (no compiled kernel recorded)")
    else:
        unread = [c["config"] for c in configs if c["equal"] is None]
        diff = [c for c in configs if c["equal"] is False]
        if unread:
            why.append(f"configs not paired or unread: {unread}")
        for c in diff:
            why.append(f"config {c['config']}: plain {c['plain']} != copy {c['copy']}")
    if sass_equal is None:
        why.append("SASS leg unread (no cuobjdump, or no compiled cubin)")
    elif not sass_equal:
        why.append("the all-off copy's SASS differs from the plain kernel's (line info stripped)")
    if hint_ok is False:
        why.append("the compiler dropped the eviction hint: no L2::cache_hint on the A/B copies in the PTX")
    return {"verdict": "FAIL" if why else "PASS", "median_dev": med, "worst_dev": worst,
            "n_ratios": len(ratios), "why": why}


def read_gate(path: Path, vid: str) -> str:
    """The gate line for `vid` in a gate.env, or '' when there is none."""
    if not Path(path).exists():
        return ""
    for line in Path(path).read_text().splitlines():
        k, _, val = line.partition("=")
        if k == f"GATE_{gate_key(vid)}":
            return val.split()[0] if val else ""
    return ""


def gate_key(vid: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in vid)


def split_gemms(ms: list[float]) -> dict:
    """A call's launches alternate w1, w2 (GEMMS_PER_CALL = 2): per-GEMM lists."""
    if len(ms) % PW.GEMMS_PER_CALL:
        raise ValueError(f"{len(ms)} launches is not whole calls of {PW.GEMMS_PER_CALL}")
    return {"w1": ms[0::2], "w2": ms[1::2]}


def stem(g: int, arm: str, n: int) -> str:
    return f"G{g}-{arm}-n{n}"


def plan_lines(variants: list[dict], mode: str) -> list[str]:
    out = [f"INSTRUMENTED KERNEL ({mode}): {len(variants)} variant(s); upstream vLLM "
           f"{I.UPSTREAM['tag']} fused_moe.py sha256 {I.UPSTREAM['file_sha256'][:16]}"]
    for v in variants:
        cells = v["gate_cells"] if mode == "perturb" else v["cells"]
        spec = v["gate_spec"] if mode == "perturb" else v["spec_obj"]
        out.append(f"  {v['id']:16s} {v['kind']:6s} {v['model']:22s} BK {v['block_k']} s{v['num_stages']} "
                   f"cells {len(cells)} spec {spec.text()}")
    if mode == "perturb":
        out.append(f"  gate (decision 2): median |plain/copy - 1| <= {100 * PERTURB_TOL['median_max']:.0f}%, "
                   f"worst <= {100 * PERTURB_TOL['worst_max']:.0f}%, identical occupancy; "
                   f"{REPS} bursts of {CALLS} calls each way")
    return out


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

def run_refusal() -> str:
    try:
        import torch
    except ImportError:
        return "no torch"
    if not torch.cuda.is_available():
        return "no CUDA device"
    import importlib.util
    if importlib.util.find_spec("vllm") is None or importlib.util.find_spec("triton") is None:
        return "vLLM or Triton is not importable here (the vLLM venv runs this)"
    return ""


def _smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unread: {exc}"


def launch_event_factory(torch):
    """THE PER-LAUNCH TIMER, a deliberate exception to timing.time_kernel (reviewed and
    flagged for the owner, rental 4). time_kernel times a whole call; the perturbation
    gate compares ONE fused_moe_kernel launch inside R3's call, plain against the copy,
    so the installer records a CUDA event pair right around each launch, the same way
    for both kernels, with the GPU held ahead of the host by a sleep (no launch waits on
    Python). It is a paired comparison on one board, never a published time."""
    return lambda: torch.cuda.Event(enable_timing=True)


def launch_ms(launches: list[dict]) -> list[float]:
    """Each recorded launch's event-pair milliseconds, in launch order."""
    return [s.elapsed_time(e) for s, e in (r["events"] for r in launches)]


class Card:
    """The box: weights per model, the call per cell, the burst timer."""

    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.override_config, self.where = SWEEP.find_override()
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe import fused_moe as FM
        self.fused_experts, self.FM = fused_experts, FM
        self.upstream = I.installed_check(FM.__file__)
        self.model = None
        self.cycles_per_ms = None

    def load(self, model: str):
        if self.model == model:
            return
        torch = self.torch
        self.w1 = self.w2 = None
        torch.cuda.empty_cache()
        cfg = MODEL_CONFIGS[model]
        self.copies, _ = PW.counter_declaration(cfg, BLOCK_M)
        self.w1, self.w2, _ = PW.build_private_weights(cfg, "bf16", self.copies, self.args.seed)
        self.nw1, self.nw2 = self.w1[::self.copies], self.w2[::self.copies]
        self.declared = {a: PW.declared_experts(a, cfg.num_experts, self.copies) for a in PW.ARMS}
        self.cfg, self.model, self.inputs = cfg, model, {}

    def call(self, v: dict, arm: str, n: int):
        key = (n, tuple(v["arms"]) if v["kind"] == "stamps" else tuple(PW.ARMS))
        if key not in self.inputs:
            self.inputs[key] = PW.arm_inputs(self.cfg, n, BLOCK_M, self.copies, self.args.seed,
                                             "bf16", self.w1.dtype, arms=key[1])
        _tokens, x, ids_by_arm, weights, kw = self.inputs[key]
        return PW.arm_call(self.fused_experts, arm, self.w1, self.w2, self.nw1, self.nw2,
                           self.declared, x, ids_by_arm, weights, kw)

    def conf(self, v: dict, g: int) -> dict:
        return dict(PW.pinned_config(BLOCK_N, g, v["num_stages"], **PW.block_k_kw(v["block_k"])),
                    BLOCK_SIZE_M=BLOCK_M)

    def sleep_ahead(self):
        torch = self.torch
        if self.cycles_per_ms is None:
            from launch_floor import ProbeGPU
            self.cycles_per_ms = ProbeGPU(torch).cycles_per_ms()
        torch.cuda._sleep(int(SLEEP_AHEAD_MS * self.cycles_per_ms))

    def flusher(self):
        if getattr(self, "_flusher", None) is None:
            from moe.bench import timing
            self._flusher = timing.L2Flusher(timing.flush_mb_for_device())
        return self._flusher

    def burst(self, fn, conf, make_ctx, calls: int = CALLS) -> tuple[list[float], list[dict]]:
        """`calls` calls under make_ctx(events) (timed_plain or install), each after an L2
        flush issued OUTSIDE the launch event pairs (R3's pages flush before every call), the
        GPU held ahead: each launch's event-pair ms in launch order, and the launch records
        (their compiled kernels). One untimed call first, so no compile lands in the burst."""
        torch = self.torch
        fl = self.flusher()
        with self.override_config(conf):
            with make_ctx(None):
                fn()
            torch.cuda.synchronize()
            with make_ctx(launch_event_factory(torch)) as rec:
                self.sleep_ahead()
                for _ in range(calls):
                    fl.flush()
                    fn()
                torch.cuda.synchronize()
        return launch_ms(rec.launches), rec.launches

    def compile_once(self, fn, conf, spec) -> list[dict]:
        """One call under the copy with `spec` (or the plain kernel, spec None), untimed: its
        launch records, for the SASS and hint legs."""
        ctx = I.timed_plain(module=self.FM) if spec is None else I.install(spec, module=self.FM)
        with self.override_config(conf):
            with ctx as rec:
                fn()
            self.torch.cuda.synchronize()
        return rec.launches


def facts_by_config(launches: list[dict], into: dict) -> dict:
    from moe.instrumented import cubin as CB
    for lr in launches:
        key = I.config_key(lr)
        if key not in into:
            f = CB.compiled_facts(lr.get("compiled"))
            if f is not None:
                into[key] = f
    return into


def sass_of(cubin: bytes | None, dst: Path | None = None) -> str | None:
    """cuobjdump -sass of a cubin's bytes (None without cuobjdump or cubin); kept at dst."""
    tool = shutil.which("cuobjdump")
    if tool is None or not cubin:
        return None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as fh:
        fh.write(cubin)
        name = fh.name
    try:
        text = subprocess.run([tool, "-sass", name], capture_output=True, text=True, timeout=120).stdout
    finally:
        os.unlink(name)
    if dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    return text or None


def key_name(key: tuple) -> str:
    return "-".join(str(k) for k in key)


def sass_equal_leg(plain: dict, off: dict, out: Path) -> tuple[bool | None, list]:
    """The plain kernel's and the all-off copy's SASS, per config, line info stripped."""
    from moe.instrumented import cubin as CB
    rows, ok = [], True
    for key in sorted(plain, key=str):
        a = sass_of(plain[key].get("cubin"), out / f"plain-{key_name(key)}.sass")
        b = sass_of((off.get(key) or {}).get("cubin"), out / f"off-{key_name(key)}.sass")
        if a is None or b is None:
            rows.append({"config": list(key), "equal": None})
            ok = None if ok is not False else ok
            continue
        eq = CB.normalize_sass(a) == CB.normalize_sass(b)
        rows.append({"config": list(key), "equal": eq, "instructions": len(CB.normalize_sass(a))})
        if not eq:
            ok = False
    return (ok if rows else None), rows


def perturb(args, variants: list[dict]) -> int:
    from moe.instrumented import cubin as CB
    card = Card(args)
    torch = card.torch
    report = {"tool": "scripts/instr_probe.py --mode perturb", "tolerance": PERTURB_TOL,
              "reps": REPS, "calls": CALLS, "upstream": card.upstream, "override_hook": card.where,
              "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
              "versions": PW._package_versions(), "device": {"name": torch.cuda.get_device_name(0),
                                                             "uuid": PW.device_identity()},
              "smi_before": _smi(), "variants": {}, "started": time.time()}
    upstream_ok = bool(card.upstream["file_matches"] and card.upstream["excerpt_matches"])
    env, worst = [], exit_codes.DONE
    for v in variants:
        card.load(v["model"])
        rows, plain_f, copy_f, off_f, hinted = [], {}, {}, {}, []
        for g, arm, n in v["gate_cells"]:
            fn, conf = card.call(v, arm, n), card.conf(v, g)
            per = {"plain": {"w1": [], "w2": []}, "copy": {"w1": [], "w2": []}}
            for rep in range(REPS):
                order = ("plain", "copy") if rep % 2 == 0 else ("copy", "plain")
                for which in order:
                    gspec = v["gate_spec"]
                    make = ((lambda ev: I.timed_plain(module=card.FM, events=ev)) if which == "plain"
                            else (lambda ev, gs=gspec: I.install(gs, module=card.FM, events=ev)))
                    ms, launches = card.burst(fn, conf, make)
                    facts_by_config(launches, plain_f if which == "plain" else copy_f)
                    ms = split_gemms(ms)
                    for gemm in ("w1", "w2"):
                        per[which][gemm].append(statistics.median(ms[gemm]))
            for gemm in ("w1", "w2"):
                p, c = statistics.median(per["plain"][gemm]), statistics.median(per["copy"][gemm])
                rows.append({"G": g, "arm": arm, "n": n, "gemm": gemm, "plain_ms": p, "copy_ms": c,
                             "ratio": p / c if c else None,
                             "plain_reps": per["plain"][gemm], "copy_reps": per["copy"][gemm]})
            # the all-off copy on the same cell (the SASS leg) and a hinted unit's own copy (the hint leg)
            facts_by_config(card.compile_once(fn, conf, I.InstrSpec()), off_f)
            if v["spec_obj"].hints:
                hinted += card.compile_once(fn, conf, v["spec_obj"])
        configs = compare_configs(plain_f, copy_f)
        out_dir = Path(args.out) / "sass" / gate_key(v["id"])
        sass_ok, sass_rows = sass_equal_leg(plain_f, off_f, out_dir)
        hint_ok = None
        if v["spec_obj"].hints:
            hf = facts_by_config(hinted, {})
            hint_ok = bool(hf) and all(CB.hint_in_ptx(f["ptx"]) for f in hf.values())
        brackets = {}
        for key, f in copy_f.items():
            t = sass_of(f.get("cubin"), out_dir / f"copy-{key_name(key)}.sass")
            if t:
                brackets[key_name(key)] = CB.sass_checks(t)
        verdict = gate_verdict([r["ratio"] for r in rows], configs, upstream_ok=upstream_ok,
                               sass_equal=sass_ok, hint_ok=hint_ok)
        report["variants"][v["id"]] = {"kind": v["kind"], "model": v["model"], "block_k": v["block_k"],
                                       "num_stages": v["num_stages"], "gate_spec": v["gate_spec"].as_dict(),
                                       "unit_spec": v["spec_obj"].as_dict(), "cells": rows,
                                       "configs": configs, "sass_equal": sass_ok, "sass_rows": sass_rows,
                                       "hint_ok": hint_ok, "copy_sass_checks_printed": brackets, **verdict}
        env.append(f"GATE_{gate_key(v['id'])}={verdict['verdict']} "
                   f"median={verdict['median_dev']} worst={verdict['worst_dev']}")
        if verdict["verdict"] != "PASS":
            worst = exit_codes.CLAIM_FAIL
        print(f"  {v['id']}: {verdict['verdict']} {'; '.join(verdict['why'])}", flush=True)
    report["smi_after"] = _smi()
    report["ended"] = time.time()
    out = Path(args.out)
    (out / "perturb.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "gate.env").write_text("\n".join(env) + "\n")
    return worst


def stamps(args, v: dict) -> int:
    import numpy as np

    from moe.instrumented import cubin as CB
    card = Card(args)
    torch = card.torch
    out = Path(args.out)
    (out / "stamps").mkdir(parents=True, exist_ok=True)
    card.load(v["model"])
    fl = card.flusher()
    index, bad, compiled = [], 0, {}
    for g, arm, n in v["cells"]:
        fn, conf = card.call(v, arm, n), card.conf(v, g)
        try:
            with card.override_config(conf):
                with I.install(v["spec_obj"], module=card.FM):
                    for _ in range(STAMP_WARMUP):
                        fn()
                torch.cuda.synchronize()
                for c in range(STAMP_CALLS):
                    fl.flush()                      # outside the stamps' launches, as R3 flushes
                    with I.install(v["spec_obj"], module=card.FM,
                                   events=launch_event_factory(torch)) as rec:
                        fn()
                        torch.cuda.synchronize()
                    facts_by_config(rec.launches, compiled)
                    ms_each = launch_ms(rec.launches)
                    for lr, gemm, lms in zip(rec.launches, ("w1", "w2"), ms_each, strict=True):
                        name = f"{stem(g, arm, n)}-call{c}-{gemm}.npy"
                        np.save(out / "stamps" / name, lr["buffer"].cpu().numpy())
                        index.append({k: lr[k] for k in lr if k not in ("buffer", "events", "compiled")}
                                     | {"G": g, "arm": arm, "n": n, "call": c, "gemm": gemm, "file": name,
                                        "launch_ms": lms, "width": v["spec_obj"].width,
                                        "marks": v["spec_obj"].marks if v["spec_obj"].stamps == 2 else 0})
        except Exception as exc:                                  # noqa: BLE001
            index.append({"G": g, "arm": arm, "n": n, "error": f"{type(exc).__name__}: {exc}"})
            bad += 1
    sass = []
    for key, f in compiled.items():
        name = f"sass/{key_name(key)}.sass"
        text = sass_of(f.get("cubin"), out / name)
        sass.append({"config": list(key), "MUL_ROUTED_WEIGHT": key[6], "file": name if text else None,
                     "checks": CB.sass_checks(text) if text else None,
                     "regs": f["regs"], "shared": f["shared"]})
    man = {"tool": "scripts/instr_probe.py --mode stamps", "variant": v["id"], "model": v["model"],
           "block_k": v["block_k"], "num_stages": v["num_stages"], "spec": v["spec_obj"].as_dict(),
           "columns": list(I.COLUMNS), "hdr": I.HDR, "upstream": card.upstream,
           "triton_cache": os.environ.get("TRITON_CACHE_DIR"), "l2_flush_mb": fl.megabytes,
           "versions": PW._package_versions(), "device": {"name": torch.cuda.get_device_name(0),
                                                          "uuid": PW.device_identity()},
           "copies_declared": card.copies, "smi": _smi(), "sass": sass, "launches": index}
    (out / "stamps.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"STAMPS {v['id']}: {len(index)} launches, {bad} failed; {out}")
    return exit_codes.INVALID if bad else exit_codes.DONE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("perturb", "stamps"), required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--variant", default=None, help="stamps: the variant id to run")
    p.add_argument("--gate", type=Path, default=None, help="stamps: the perturb unit's gate.env")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        variants = load_variants(args.variants)
        if args.mode == "stamps":
            mine = [v for v in variants if v["id"] == args.variant]
            if len(mine) != 1:
                raise Refused(f"--variant {args.variant!r} is not one of {[v['id'] for v in variants]}")
            if mine[0]["kind"] != "stamps":
                raise Refused(f"variant {args.variant} is a {mine[0]['kind']} unit, not stamps")
            variants = mine
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    print("\n".join(plan_lines(variants, args.mode)))
    if args.mode == "stamps":
        g = read_gate(args.gate, args.variant) if args.gate else ""
        if g != "PASS":
            print(f"REFUSED: the perturbation gate for {args.variant} reads {g or 'nothing'} "
                  f"({args.gate}): an instrumented unit runs only after its gate PASSES")
            return exit_codes.REFUSED
    if args.dry_run:
        print("DRY RUN: nothing launched (exit 2)")
        return exit_codes.REFUSED
    why = run_refusal()
    if why:
        print(f"REFUSED: {why}")
        return exit_codes.REFUSED
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # a cache of this unit's own (build-r4-review bug G1: env.sh's shared cache let an earlier
    # unit's compile hide this one's); the driver sets the same directory
    os.environ["TRITON_CACHE_DIR"] = str(Path(args.out) / "triton-cache")
    return perturb(args, variants) if args.mode == "perturb" else stamps(args, variants[0])


if __name__ == "__main__":
    sys.exit(main())
