#!/usr/bin/env python3
"""Can the R3 counter run happen on THIS box? The VM's preflight, never cached.

    . ~/moe/env.sh && python3 ~/moe/repo/scripts/vm_preflight.py

scripts/setup_vm.sh runs this as its last stage (S8) and after every
`--check`/`--preflight-only`. It writes `$SESSION_ROOT/PREFLIGHT.txt` and
`PREFLIGHT.json`, one PASS/FAIL/SKIP line per check, prints one `RESULT:` line
per check (moe/bench/exit_codes.py's format, every check a CLAIM: "this box
can take the measurement"), and exits through `exit_codes.classify`: 0 only
when every check PASSED, 1 otherwise, 4 on a crash. A SKIP is a check not run
because an earlier one already failed, and it counts against the box.

  PF1 card    nvidia-smi and torch in BOTH venvs name one card (name and
              UUID); capability, SMs and L2 are recorded, and the page says
              when this is not the study's H200.
  PF2 wheel   each venv's torch.version.cuda major is at most the driver's
              CUDA major, and a kernel launch succeeds: a PyPI cu13 torch in
              the vLLM venv on a pre-r580 driver fails here, not mid-session.
  PF3 stack   base torch and triton, and the vLLM venv's vllm, torch and
              triton, are the versions requirements/resolved-*.txt pin (read,
              never typed), and vLLM's fused_experts imports.
  PF4 metrics `ncu --query-metrics` (and the launch collection, for
              launch__*) lists every STRICT metric the r3-arms family asks;
              absent RECORDED metrics are dropped and listed, absent
              CROSS-CHECK metrics leave their gate unasked. When ncu's output
              names its lock file, that is the cause PF4 reports.
  PF5 probe   `dram_counter_route.py --probe --family r3-arms` under the
              counter door reads OPEN, exit 0, and its RESULT lines agree,
              and its per-metric record (`metrics_proven`) holds every
              metric of the family's STRICT class: a counter was actually
              READ, and so was every metric the run refuses without. THE
              GATE. Nothing else here can pass a box on which no counter
              came back, and a payload with no per-metric record fails.
  PF6 census  `dram_counter_route.py --run --family r3-arms --census-only`
              exits 0 (GEMMS_PER_CALL = 2, grids match, memory plan fits).
  PF7 RAM     when the card cannot hold ncu's first-pass save beside R3's
              allocation, host RAM holds 1.5x the footprint.

THE r3-arms FAMILY IS dram_counter_route.py's, not this file's. PF4 and PF5
read its metric classes (`R3_STRICT_METRICS`, `R3_CROSSCHECK_METRICS`,
`R3_RECORDED_METRICS`) and PF5/PF6 run its CLI; a checkout whose
dram_counter_route.py has no `--family` fails all three, saying so. PF7's
footprint is R3's own `memory_plan` at R3's own defaults with no flush buffer
(the counter child times nothing), not a number typed here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from moe.bench import exit_codes  # noqa: E402  (torch-free)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
#: The family the R3 counter run is registered under in dram_counter_route.py.
FAMILY = "r3-arms"
#: The three metric classes PF4 and PF5 read off dram_counter_route.py, in the
#: order STRICT (the run refuses without them), CROSS-CHECK, RECORDED.
METRIC_CLASSES = ("R3_STRICT_METRICS", "R3_CROSSCHECK_METRICS", "R3_RECORDED_METRICS")
#: The card every timed page of the study was measured on (H200 sessions 1-5).
#: A page from any other card is that card's, and says so.
STUDY_CARD = "nvidia_h200"
#: Host RAM against the footprint when ncu's first-pass save spills to the host.
HOST_SAVE_FACTOR = 1.5
PROBE_TIMEOUT_S = 600
CENSUS_TIMEOUT_S = 1800
TORCH_TIMEOUT_S = 600
MARKER = "VMPF "

#: Run under each venv's interpreter: what that venv's torch (and vLLM) see.
#: The UUID is R3's own `device_identity`, so the card is named one way here
#: and in every R3 page.
TORCH_PROBE = r"""
import json, sys
repo, kind = sys.argv[1], sys.argv[2]
sys.path[:0] = [repo, repo + "/scripts"]
out = {}
def err(e):
    return (type(e).__name__ + ": " + str(e))[:300]
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    try:
        import triton
        out["triton"] = triton.__version__
    except Exception as e:
        out["triton_error"] = err(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out.update(name=p.name, capability="%d.%d" % (p.major, p.minor),
                   sm_count=p.multi_processor_count,
                   l2_bytes=int(getattr(p, "L2_cache_size", 0) or 0),
                   memory_bytes=int(p.total_memory))
        try:
            import private_weight_reference as PWR
            out["uuid"] = PWR.device_identity()
        except Exception as e:
            out["uuid_error"] = err(e)
        try:
            x = torch.arange(1024, device="cuda", dtype=torch.float32)
            out["kernel_ok"] = float((x + 1).sum().item()) == 524800.0
            torch.cuda.synchronize()
        except Exception as e:
            out["kernel_ok"] = False
            out["kernel_error"] = err(e)
    else:
        out["cuda_error"] = "torch.cuda.is_available() is False"
except Exception as e:
    out["torch_error"] = err(e)
if kind == "vllm":
    try:
        import vllm
        out["vllm"] = vllm.__version__
    except Exception as e:
        out["vllm_error"] = err(e)
    try:
        from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: F401
        out["fused_experts"] = True
    except Exception as e:
        out["fused_experts"] = False
        out["fused_experts_error"] = err(e)
print("VMPF " + json.dumps(out), flush=True)
"""


@dataclass
class Check:
    id: str
    what: str
    verdict: str
    detail: str
    data: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.id} {self.verdict} {self.what}: {self.detail}"

    def scored(self) -> str:
        """The shared table's spelling: a SKIP is UNKNOWN, which counts against."""
        return {PASS: exit_codes.PASS, FAIL: exit_codes.FAIL}.get(self.verdict,
                                                                 exit_codes.UNKNOWN)

    def result(self) -> str:
        return exit_codes.result_line(exit_codes.CLAIM, self.id, self.scored(),
                                      " ".join(f"{self.what}: {self.detail}".split()))


def _one(text: str) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------------
# reading the box: pure parsers
# --------------------------------------------------------------------------

def parse_smi(query: str, header: str) -> dict | None:
    """nvidia-smi's first card, from `--query-gpu=name,uuid,driver_version,
    memory.total,memory.free --format=csv,noheader,nounits` and the plain
    header's `CUDA Version: X.Y`. None when no card is listed."""
    rows = [r for r in query.splitlines() if r.strip()]
    if not rows:
        return None
    parts = [p.strip() for p in rows[0].split(",")]
    if len(parts) < 5:
        return None
    m = re.search(r"CUDA Version:\s*([0-9.]+)", header or "")

    def mib(v: str) -> int | None:
        return int(float(v)) * 2**20 if re.fullmatch(r"[0-9.]+", v) else None

    return {"name": parts[0], "uuid": parts[1], "driver": parts[2],
            "memory_bytes": mib(parts[3]), "free_bytes": mib(parts[4]),
            "cuda": m.group(1) if m else None, "count": len(rows)}


def bare_uuid(text: str | None) -> str:
    """One spelling of a card's UUID: the chain's own normaliser."""
    from alpha_g_chain_helpers import _bare_uuid
    return _bare_uuid(text or "")


def major(version: str | None) -> int | None:
    m = re.match(r"\s*(\d+)", version or "")
    return int(m.group(1)) if m else None


def public(version: str | None) -> str:
    """A version without its local tag: 2.13.0+cu130 -> 2.13.0."""
    return (version or "").split("+", 1)[0].strip()


def resolved_pins(req_dir: Path) -> dict[str, dict[str, str]]:
    """What PF3 asks of each venv, off requirements/resolved-*.txt: base's
    torch and triton, the vLLM venv's vllm, torch and triton."""
    want = {"base": ("torch", "triton"), "vllm": ("vllm", "torch", "triton")}
    out: dict[str, dict[str, str]] = {}
    for env, names in want.items():
        pins: dict[str, str] = {}
        path = req_dir / f"resolved-{env}.txt"
        if path.is_file():
            for raw in path.read_text().splitlines():
                line = raw.split("#", 1)[0].strip()
                if "==" in line:
                    name, ver = line.split("==", 1)
                    if name.strip().lower() in names:
                        pins[name.strip().lower()] = public(ver)
        out[env] = pins
    return out


def metric_names(text: str) -> set[str]:
    """Every metric base name an `ncu --query-metrics` listing mentions."""
    return set(re.findall(r"\b([a-z][a-z0-9]*__[a-z0-9_]+)\b", text or ""))


#: A line of ncu's output about its lock file (docs/LAMBDA.md: under /tmp,
#: owned by the first user that ran ncu).
LOCK_FILE_LINE = re.compile(r"lock ?file|nsight-compute-lock", re.I)


def lock_file_line(text: str) -> str:
    """The first line of ncu's output that names its lock file, or ""."""
    return next((_one(ln) for ln in (text or "").splitlines() if LOCK_FILE_LINE.search(ln)), "")


def base_name(metric: str) -> str:
    """`dram__bytes_read.sum` -> `dram__bytes_read`, as --query-metrics lists it."""
    return metric.split(".", 1)[0]


def implied_code(log_text: str) -> int | None:
    """The exit code a log's RESULT lines imply, or None when it printed none."""
    try:
        return exit_codes.classify_text(log_text)
    except exit_codes.NoGatesScored:
        return None


# --------------------------------------------------------------------------
# the checks: pure functions of what was read
# --------------------------------------------------------------------------

def card_line(t: dict, slug: str) -> str:
    l2 = t.get("l2_bytes") or 0
    return (f"CARD {t.get('name')} ({slug}, UUID {bare_uuid(t.get('uuid'))}, "
            f"sm_{str(t.get('capability', '?')).replace('.', '')}, {t.get('sm_count')} SMs, "
            f"{l2 / 2**20:.0f} MiB L2): every number here is THIS card's; the study's timing "
            f"pages are {STUDY_CARD}.")


def pf1_card(smi: dict | None, base: dict, vllm: dict) -> Check:
    what = "nvidia-smi and torch in both venvs name one card"
    if smi is None:
        return Check("PF1", what, FAIL, "nvidia-smi lists no card")
    names = {"nvidia-smi": smi.get("name"), "base": base.get("name"),
             "vllm": vllm.get("name")}
    uuids = {"nvidia-smi": bare_uuid(smi.get("uuid")), "base": bare_uuid(base.get("uuid")),
             "vllm": bare_uuid(vllm.get("uuid"))}
    views = {"base": base, "vllm": vllm}
    missing = [k for k in views if not names[k]]
    if missing:
        why = "; ".join(f"{k}: " + (views[k].get("torch_error") or views[k].get("cuda_error")
                                    or "no device") for k in missing)
        return Check("PF1", what, FAIL, f"torch saw no card in {', '.join(missing)} ({why})",
                     {"names": names, "uuids": uuids})
    if len(set(names.values())) != 1 or len(set(uuids.values())) != 1:
        return Check("PF1", what, FAIL, f"they disagree: names {names}, uuids {uuids}",
                     {"names": names, "uuids": uuids})
    try:
        from moe.bench.provenance import card_slug
        slug = card_slug(str(base["name"]))
    except Exception as exc:                                          # noqa: BLE001
        return Check("PF1", what, FAIL, f"the card has no slug: {exc}")
    data = {"name": base["name"], "slug": slug, "uuid": uuids["base"],
            "capability": base.get("capability"), "sm_count": base.get("sm_count"),
            "l2_bytes": base.get("l2_bytes"), "memory_bytes": base.get("memory_bytes"),
            "driver": smi.get("driver"), "study_card": STUDY_CARD,
            "same_card_as_study": slug == STUDY_CARD, "card_line": card_line(base, slug)}
    note = ("" if slug == STUDY_CARD
            else f"; NOT the study's {STUDY_CARD}: every alpha here is this card's")
    return Check("PF1", what, PASS, f"{base['name']} ({slug}), UUID {uuids['base']}{note}", data)


def pf2_wheel(smi: dict | None, base: dict, vllm: dict) -> Check:
    what = "each venv's torch CUDA major <= the driver's, and a kernel launches"
    driver_cuda = major((smi or {}).get("cuda"))
    if driver_cuda is None:
        return Check("PF2", what, FAIL, "nvidia-smi's header names no CUDA version")
    bad, seen = [], {}
    for env, t in (("base", base), ("vllm", vllm)):
        wheel = major(t.get("cuda"))
        seen[env] = {"torch_cuda": t.get("cuda"), "kernel_ok": t.get("kernel_ok")}
        if wheel is None:
            why = t.get("torch_error") or t.get("cuda") or "cpu build"
            bad.append(f"{env}: no CUDA torch ({why})")
        elif wheel > driver_cuda:
            bad.append(f"{env}: torch built for CUDA {t.get('cuda')} on a driver serving CUDA "
                       f"{smi.get('cuda')}")
        elif t.get("kernel_ok") is not True:
            bad.append(f"{env}: the kernel launch failed "
                       f"({t.get('kernel_error') or t.get('cuda_error') or 'no device'})")
    data = {"driver_cuda": smi.get("cuda"), "venvs": seen}
    if bad:
        return Check("PF2", what, FAIL, "; ".join(bad), data)
    return Check("PF2", what, PASS, f"driver CUDA {smi.get('cuda')}; base {base.get('cuda')}, "
                 f"vllm {vllm.get('cuda')}; both launched", data)


def pf3_stack(base: dict, vllm: dict, pins: dict) -> Check:
    what = "the venvs hold the versions requirements/resolved-*.txt pin, and fused_experts imports"
    got = {"base": base, "vllm": vllm}
    bad = []
    for env, want in pins.items():
        if not want:
            bad.append(f"requirements/resolved-{env}.txt pins none of what PF3 checks")
        for name, ver in sorted(want.items()):
            have = public(got[env].get(name))
            if have != ver:
                bad.append(f"{env} {name} {have or 'absent'} (pinned {ver})")
    if vllm.get("fused_experts") is not True:
        why = vllm.get("fused_experts_error") or vllm.get("vllm_error") or "not tried"
        bad.append(f"vllm's fused_experts does not import ({why})")
    data = {"pins": pins, "have": {e: {k: got[e].get(k) for k in ("torch", "triton", "vllm")}
                                   for e in got}}
    if bad:
        return Check("PF3", what, FAIL, "; ".join(bad), data)
    return Check("PF3", what, PASS, "; ".join(f"{e}: " + ", ".join(f"{k} {v}" for k, v in
                                                                   sorted(w.items()))
                                               for e, w in pins.items()), data)


def family_metric_classes(module) -> dict[str, tuple[str, ...]] | None:
    """The r3-arms family's three metric classes, off dram_counter_route.py, or
    None when this checkout registers none."""
    got = [getattr(module, name, None) for name in METRIC_CLASSES]
    if got[0] is None:
        return None
    return {"strict": tuple(got[0]), "crosscheck": tuple(got[1] or ()),
            "recorded": tuple(got[2] or ())}


def pf4_metrics(classes: dict | None, profiling: str, launch: str, *,
                ncu: str | None) -> Check:
    what = "ncu lists every STRICT metric the r3-arms family asks"
    if not ncu:
        return Check("PF4", what, FAIL, "no ncu on PATH (setup_vm.sh S5 puts it first)")
    if classes is None:
        return Check("PF4", what, FAIL, "this checkout's dram_counter_route.py registers no "
                     f"{FAMILY} metric classes ({', '.join(METRIC_CLASSES)}), so the R3 counter "
                     "run cannot happen from it")
    listed = {"profiling": metric_names(profiling), "launch": metric_names(launch)}

    def absent(names):
        return [m for m in names if base_name(m) not in
                listed["launch" if m.startswith("launch__") else "profiling"]]

    strict, cross, recorded = (absent(classes[k]) for k in ("strict", "crosscheck", "recorded"))
    data = {"strict_absent": strict, "crosscheck_absent": cross, "recorded_dropped": recorded,
            "listed": {k: len(v) for k, v in listed.items()}}
    lock = lock_file_line(profiling + "\n" + launch)
    if lock:
        data["lock_file"] = lock
        return Check("PF4", what, FAIL, f"ncu could not take its lock file ({lock}). The lock "
                     "under /tmp belongs to the first user that ran ncu, and an ncu run as the "
                     "login user and another as root refuse each other: remove the file ncu "
                     "names and run every ncu through the counter door's launcher", data)
    if not listed["profiling"]:
        return Check("PF4", what, FAIL, "ncu --query-metrics listed no metric at all", data)
    if strict:
        return Check("PF4", what, FAIL, f"STRICT metric(s) not listed for this card: "
                     f"{', '.join(strict)}", data)
    notes = [f"{len(classes['strict'])} STRICT listed"]
    if cross:
        notes.append(f"cross-check gates NOT asked for {', '.join(cross)}")
    if recorded:
        notes.append(f"recorded metrics dropped: {', '.join(recorded)}")
    return Check("PF4", what, PASS, "; ".join(notes), data)


def family_supported(help_text: str) -> bool:
    return "--family" in (help_text or "")


def pf5_probe(rc: int | None, payload: dict | None, log_text: str, *,
              strict: tuple[str, ...] | None, why_not_run: str = "") -> Check:
    """PASS iff every STRICT metric came back: exit 0, verdict OPEN, the route
    open by dram_counter_route's own `counter_route_is_open`, the log's RESULT
    lines imply the same exit code the process returned, AND the probe's
    per-metric record (`metrics_proven`, the metrics that came back as numbers
    on one profiled launch) holds every metric in `strict`, the family's
    STRICT class. One counter read is not the run's metrics read: the run
    refuses a page whose STRICT metric is not numeric, so a box that proved
    only `dram__bytes_read.sum` would be called READY and refuse on paid
    time. A payload with no per-metric record fails, so a probe that did not
    record one cannot pass."""
    what = (f"dram_counter_route.py --probe --family {FAMILY} READ every STRICT metric "
            "here")
    if why_not_run:
        return Check("PF5", what, FAIL, why_not_run)
    from dram_counter_route import counter_route_is_open
    ncu = (payload or {}).get("ncu") or {}
    implied = implied_code(log_text)
    data = {"rc": rc, "verdict": (payload or {}).get("verdict"), "implied": implied,
            "cause": ncu.get("cause"), "version": ncu.get("version")}
    for key in ("csv_layout", "csv_header", "metrics_asked", "metrics_dropped"):
        if key in ncu:
            data[key] = ncu[key]
    if payload is None:
        return Check("PF5", what, FAIL, f"the probe wrote no payload (exit {rc})", data)
    if implied is not None and implied != rc:
        return Check("PF5", what, FAIL, f"DEFECT: the probe exited {rc} and its RESULT lines "
                     f"imply {implied}", data)
    if rc != exit_codes.DONE or payload.get("verdict") != "OPEN" or not counter_route_is_open(ncu):
        return Check("PF5", what, FAIL, f"verdict {payload.get('verdict')}, exit {rc}: "
                     f"{_one(ncu.get('cause') or ncu.get('why') or 'no cause recorded')}", data)
    if not strict:
        return Check("PF5", what, FAIL, f"this checkout's dram_counter_route.py registers no "
                     f"STRICT metric class for {FAMILY}, so there is nothing to hold the "
                     "probe's reading to", data)
    proven = ncu.get("metrics_proven")
    data["metrics_proven"] = proven
    if not isinstance(proven, list):
        return Check("PF5", what, FAIL, "the probe read a counter but recorded no per-metric "
                     "reading (metrics_proven), so it has not shown that every STRICT metric "
                     f"came back as a number: {_one(ncu.get('cause') or 'OPEN')}", data)
    unread = [m for m in strict if m not in proven]
    data["strict_unread"] = unread
    if unread:
        return Check("PF5", what, FAIL, f"OPEN, but STRICT metric(s) not read back as numbers "
                     f"by the probe: {', '.join(unread)}; the run would refuse its pages on "
                     "them", data)
    return Check("PF5", what, PASS, f"all {len(strict)} STRICT metrics read back: "
                 f"{_one(ncu.get('cause') or 'OPEN')}", data)


def pf6_census(rc: int | None, wrote: bool, log_text: str, *, skip_why: str = "") -> Check:
    what = f"dram_counter_route.py --run --family {FAMILY} --census-only passes"
    if skip_why:
        return Check("PF6", what, SKIP, skip_why)
    implied = implied_code(log_text)
    data = {"rc": rc, "implied": implied, "wrote": wrote}
    if implied is not None and implied != rc:
        return Check("PF6", what, FAIL, f"DEFECT: the census exited {rc} and its RESULT lines "
                     f"imply {implied}", data)
    if rc != exit_codes.DONE or not wrote:
        last = next((ln for ln in reversed(log_text.splitlines()) if ln.strip()), "an empty log")
        return Check("PF6", what, FAIL, f"exit {rc}, census {'written' if wrote else 'NOT written'}"
                     f"; its log ends: {_one(last)[:200]}", data)
    return Check("PF6", what, PASS, "GEMMS_PER_CALL, grids and the memory plan as the census "
                 "gates them", data)


def exit_for(checks: list[Check]) -> int:
    """The preflight's exit code, through the shared table: every check a
    CLAIM, a SKIP spelled UNKNOWN (it counts against the box), so DONE (0)
    needs every check to say PASS in so many words."""
    return exit_codes.classify((exit_codes.CLAIM, c.scored()) for c in checks)


def r3_footprint() -> tuple[int, str]:
    """What R3's counter child allocates: `memory_plan` at R3's own defaults
    (model, dtype, BLOCK_M, treads, the declared copies), no flush buffer."""
    import block_m_crossing_sweep as SWEEP
    import private_weight_reference as PWR

    from moe.spec import MODEL_CONFIGS, dtype_bytes
    args = PWR.build_parser().parse_args([])
    cfg = MODEL_CONFIGS[args.model]
    treads = PWR.ladder_treads(cfg, args.block_m, args.treads)
    copies, _why = PWR.declared_copies_for(cfg, treads, args.block_m)
    tokens = SWEEP.tokens_for_rows(cfg, treads[-1] * args.block_m)
    plan = PWR.memory_plan(cfg, args.dtype, dtype_bytes(args.dtype), copies, tokens, None,
                           "not read by the preflight",
                           flush=(0, "the counter child times nothing, so no flush buffer"))
    return plan.predicted_peak_bytes, (f"R3 memory_plan: {args.model} {args.dtype}, BLOCK_M "
                                       f"{args.block_m}, treads {treads}, {copies} copies")


def pf7_host_ram(free_bytes: int | None, footprint: int | None, mem_available: int | None,
                 *, source: str = "") -> Check:
    what = "ncu's first-pass save has somewhere to go"
    gb = 1e9
    if footprint is None or free_bytes is None:
        return Check("PF7", what, FAIL, f"footprint {footprint} or device free {free_bytes} unread")
    data = {"footprint_bytes": footprint, "device_free_bytes": free_bytes,
            "mem_available_bytes": mem_available, "factor": HOST_SAVE_FACTOR, "source": source}
    left = free_bytes - footprint
    if left >= footprint:
        return Check("PF7", what, PASS, f"the card keeps {left / gb:.1f} GB free beside the "
                     f"{footprint / gb:.1f} GB footprint, so the save stays on the device", data)
    need = HOST_SAVE_FACTOR * footprint
    if mem_available is None:
        return Check("PF7", what, FAIL, f"the save spills to the host ({left / gb:.1f} GB left "
                     f"on the card) and host RAM is unread", data)
    if mem_available >= need:
        return Check("PF7", what, PASS, f"the save spills to the host; {mem_available / gb:.0f} GB "
                     f"available >= {HOST_SAVE_FACTOR} x {footprint / gb:.1f} GB", data)
    return Check("PF7", what, FAIL, f"the save spills to the host and only {mem_available / gb:.0f}"
                 f" GB is available, under {HOST_SAVE_FACTOR} x {footprint / gb:.1f} GB", data)


# --------------------------------------------------------------------------
# running things
# --------------------------------------------------------------------------

def run(argv: list[str], timeout: float, cwd: Path = ROOT) -> tuple[int, str]:
    """One child, bounded, its process group killed on a timeout: the
    counter route's own `_run`, not a second copy of it."""
    from dram_counter_route import _run
    here = os.getcwd()
    try:
        os.chdir(cwd)
        rc, out, err = _run(argv, timeout=timeout)
    finally:
        os.chdir(here)
    return rc, (out or "") + (err or "")


def torch_view(py: str, kind: str) -> dict:
    if not py or not Path(py).exists():
        return {"torch_error": f"no interpreter at {py or '(unset)'}"}
    rc, text = run([py, "-c", TORCH_PROBE, str(ROOT), kind], TORCH_TIMEOUT_S)
    for line in reversed(text.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                break
    return {"torch_error": f"exit {rc}, no {MARKER.strip()} line: {_one(text[-300:])}"}


def mem_available(path: Path) -> int | None:
    try:
        m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", path.read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) * 1024 if m else None


def git_state() -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    home = Path(os.environ.get("MOE_HOME", Path.home() / "moe"))
    ap.add_argument("--session", default=os.environ.get("SESSION_ROOT", str(home / "session")))
    ap.add_argument("--py-base",
                    default=os.environ.get("PY_BASE", str(home / "venvs/base/bin/python")))
    ap.add_argument("--py-vllm",
                    default=os.environ.get("PY_VLLM", str(home / "venvs/vllm/bin/python")))
    ap.add_argument("--launcher", default=os.environ.get("MOE_COUNTER_LAUNCHER", ""))
    ap.add_argument("--door", default=os.environ.get("MOE_COUNTER_DOOR", ""))
    ap.add_argument("--meminfo", default=os.environ.get("MOE_MEMINFO", "/proc/meminfo"))
    args = ap.parse_args(argv)
    session = Path(args.session)
    (session / "logs").mkdir(parents=True, exist_ok=True)
    launcher = shlex.split(args.launcher)
    route = str(ROOT / "scripts" / "dram_counter_route.py")

    smi = None
    if shutil.which("nvidia-smi"):
        _, q = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits"], 60)
        _, header = run(["nvidia-smi"], 60)
        smi = parse_smi(q, header)
    base, vllm = torch_view(args.py_base, "base"), torch_view(args.py_vllm, "vllm")
    checks = [pf1_card(smi, base, vllm), pf2_wheel(smi, base, vllm),
              pf3_stack(base, vllm, resolved_pins(ROOT / "requirements"))]

    ncu = shutil.which("ncu")
    import dram_counter_route as D
    profiling = launch = ""
    if ncu:
        _, profiling = run([*launcher, ncu, "--query-metrics"], 300)
        _, launch = run([*launcher, ncu, "--query-metrics-collection", "launch"], 300)
        (session / "ncu-query-metrics.txt").write_text(profiling + "\n--- launch\n" + launch)
    classes = family_metric_classes(D)
    checks.append(pf4_metrics(classes, profiling, launch, ncu=ncu))

    why = ""
    if not ncu:
        why = "no ncu on PATH"
    elif not Path(args.py_vllm).exists():
        why = f"no vLLM interpreter at {args.py_vllm}"
    else:
        _, help_text = run([args.py_vllm, route, "--help"], 120)
        if not family_supported(help_text):
            why = (f"this checkout's dram_counter_route.py has no --family, so it cannot run the "
                   f"{FAMILY} probe; a counter reading from the ladder family's probe would not "
                   "prove the r3-arms metrics readable")
    rc = payload = None
    log_text = ""
    if not why:
        out = session / "probe.json"
        out.unlink(missing_ok=True)
        rc, log_text = run([*launcher, args.py_vllm, route, "--probe", "--family", FAMILY,
                            "--out", str(out)], PROBE_TIMEOUT_S)
        (session / "logs" / "pf5-probe.log").write_text(log_text)
        try:
            payload = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            payload = None
    checks.append(pf5_probe(rc, payload, log_text, why_not_run=why,
                            strict=classes["strict"] if classes else None))

    if checks[-1].verdict != PASS:
        checks.append(pf6_census(None, False, "", skip_why="PF5 did not read a counter, so a "
                                 "census under ncu would test nothing"))
    else:
        out = session / "census.json"
        out.unlink(missing_ok=True)
        crc, clog = run([*launcher, args.py_vllm, route, "--run", "--family", FAMILY,
                         "--census-only", "--out", str(out)], CENSUS_TIMEOUT_S)
        (session / "logs" / "pf6-census.log").write_text(clog)
        checks.append(pf6_census(crc, out.is_file(), clog))

    try:
        footprint, source = r3_footprint()
    except Exception as exc:                                          # noqa: BLE001
        footprint, source = None, f"R3's memory plan raised {type(exc).__name__}: {exc}"
    checks.append(pf7_host_ram((smi or {}).get("free_bytes"), footprint,
                               mem_available(Path(args.meminfo)), source=source))

    card = checks[0].data
    code = exit_for(checks)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = [f"THE VM PREFLIGHT, {stamp}: can the R3 counter run happen on this box",
            card.get("card_line") or f"CARD {(smi or {}).get('name', 'none')}: PF1 did not name "
            "one card, so nothing measured here can be attributed to a card yet",
            f"commit {git_state()['commit']}{' (DIRTY)' if git_state()['dirty'] else ''}; "
            f"counter door {args.door or 'unset'}; launcher {args.launcher or '(none)'}; "
            f"ncu {ncu or 'none'}"]
    lines = head + [c.line() for c in checks] + [
        f"verdict {'READY' if code == exit_codes.DONE else 'NOT READY'} "
        f"(exit {code}: {exit_codes.describe(code)})"]
    (session / "PREFLIGHT.txt").write_text("\n".join(lines) + "\n")
    (session / "PREFLIGHT.json").write_text(json.dumps({
        "schema": 1, "utc": stamp, "git": git_state(), "door": args.door,
        "launcher": args.launcher, "ncu": ncu, "card": card, "smi": smi,
        "venvs": {"base": base, "vllm": vllm},
        "checks": [asdict(c) for c in checks], "all_pass": code == exit_codes.DONE,
        "exit": code}, indent=2, default=str) + "\n")
    for line in lines:
        print(line)
    for c in checks:
        print(c.result())
    print(f"wrote {session / 'PREFLIGHT.txt'} and PREFLIGHT.json")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:                                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"vm_preflight crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(exit_codes.ERROR) from exc
