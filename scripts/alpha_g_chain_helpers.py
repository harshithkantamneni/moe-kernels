#!/usr/bin/env python
"""The Python half of scripts/alpha_g_chain.sh: every read of a report.json, a
log or a ledger that the chain needs, as one small command each, so the shell
stays a sequencer and the reading is testable off GPU.

    alpha_g_chain_helpers.py card                   -> card slug, why not (the probe's reason)
    alpha_g_chain_helpers.py device                 -> this card's UUID, as R3's DEVICE file keys it
    alpha_g_chain_helpers.py run-id LOG EXPERIMENT  -> the run id the plan page printed
    alpha_g_chain_helpers.py estimate LOG           -> seconds the arm's own --dry-run priced
    alpha_g_chain_helpers.py reading REPORT         -> one PAIRS row's ratio half (R3)
    alpha_g_chain_helpers.py pairs REPORT...        -> the reports that FORMED a ratio, one per line
    alpha_g_chain_helpers.py eta REPORT             -> eta, lo, hi, band, exit word (R1)
    alpha_g_chain_helpers.py eta-for SESSION RESULTS G  -> the same, for the G's R1 log
    alpha_g_chain_helpers.py gate REPORT TAG        -> that gate's verdict, `absent` or `unreadable`
    alpha_g_chain_helpers.py verdict LOG            -> what its RESULT lines imply (2nd opinion)
    alpha_g_chain_helpers.py pairs-table SESSION RESULTS LADDER SEEDS [key=value...]
                                                    -> rewrites PAIRS.tsv and PAIRS-fixed.tsv
    alpha_g_chain_helpers.py calibration-dir LOG    -> the run directory calibrate wrote

Every command prints tab-separated fields on ONE line (or one path per line for
`pairs`) and exits 0; a report that cannot be read prints `unreadable` in the
first field rather than crashing the chain.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from moe.bench import exit_codes  # noqa: E402

RUN_ID_LINE = re.compile(r"^experiment\s+(\S+)\s*/\s*(\S+)\s*$", re.M)
ESTIMATE_LINES = (
    re.compile(r"takes about (\d+) s"),                  # R3 at a duty below 1: the wall figure
    re.compile(r"estimated GPU time (\d+) s"),           # R3 at full duty
    re.compile(r"estimated wall time (\d+) s"),          # R1
)
#: The one line calibrate_hardware.py prints for the ruler it measured, under
#: its untracked run directory (the published copy is a second line, PUBLISHED).
CALIBRATE_WROTE = re.compile(r"^\[calibrate\] wrote (\S+\.yaml)\s*$", re.M)

#: The R1 exit words a regime word may be read off. INVALID is a page whose
#: own validity gates refused it (R1's V7 says an interval above 1 means
#: "something other than the SM clock moved"), and REFUSED, ERROR and an
#: unscored page scored nothing: the word is withheld, never computed.
BAND_QUOTABLE = (exit_codes.CODE_NAMES[exit_codes.DONE],
                 exit_codes.CODE_NAMES[exit_codes.CLAIM_FAIL])
#: What an R1 column reads when the G has no R1 report on disk yet.
ETA_UNMEASURED = ["none", "none", "none", "unmeasured", "unscored"]


def _load(path: str | Path) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _fmt(v, spec: str = ".4f") -> str:
    return "none" if v is None else format(float(v), spec)


def _one_line(text: str) -> str:
    """No tab and no newline: the field goes into a TSV row or a ledger note."""
    return " ".join(str(text).split())


def _exit_word(payload: dict) -> str:
    gates = payload.get("gates") or []
    if not gates:
        return "unscored"
    try:
        rc = exit_codes.classify((g["kind"], g.get("tag") or g.get("number"), g["verdict"])
                                 for g in gates)
    except Exception:                                             # noqa: BLE001
        return "unscored"
    return exit_codes.CODE_NAMES.get(rc, str(rc))


def card() -> list[str]:
    """`slug  reason`: the card as the driver and every report name it, through
    moe/bench/provenance, and why not when it cannot be named. The reason is
    the probe's own text, because "nocard" alone hid a torch wheel built for a
    newer driver than the host's behind a STOP that named the wrong row."""
    try:
        from moe.bench.provenance import card_slug, provenance_block
    except Exception as exc:                                      # noqa: BLE001
        return ["nocard", _one_line(f"moe.bench.provenance is not importable from "
                                    f"{ROOT}: {exc.__class__.__name__}: {exc}")]
    try:
        prov = provenance_block(repo_root=ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return ["nocard", _one_line(f"provenance_block raised {exc.__class__.__name__}: {exc}")]
    if not prov.gpu_name:
        return ["nocard", _one_line(prov.missing.get("gpu_name", "no reason recorded"))]
    return [card_slug(prov.gpu_name), ""]


def _bare_uuid(text: str) -> str:
    """torch prints a card's UUID bare and nvidia-smi prefixes it `GPU-`: one
    spelling, so a pass that read one source and a resume that read the other
    still agree on the card."""
    text = text.strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def device() -> str:
    """This card's identity: R3's own `device_identity` (torch's UUID, so the
    chain's DEVICE and the ratio arm's DEVICE name the card one way), then
    nvidia-smi's when torch gave none, else whatever weaker identity R3 would
    record, else ''."""
    import private_weight_reference as PWR
    ident = PWR.device_identity()
    if ident and not ident.startswith(PWR.NO_UUID_PREFIX):
        return _bare_uuid(ident)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            out = None
        if out is not None and out.returncode == 0 and out.stdout.strip():
            return _bare_uuid(out.stdout.strip().splitlines()[0])
    return ident


def run_id(log: str | Path, experiment: str) -> str:
    """The run id the arm's plan page printed for `experiment`, or ''."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    for name, rid in RUN_ID_LINE.findall(text):
        if name == experiment:
            return rid
    return ""


def estimate(log: str) -> str:
    """Seconds the arm's own --dry-run priced, off its plan page; '' if none."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    for pattern in ESTIMATE_LINES:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return ""


def reading_header() -> list[str]:
    """The ratio half of PAIRS.tsv's header, the arm names read from the arm."""
    import private_weight_reference as PWR
    return ["G", "seed", "ratio", "lo", "hi", "exit", "duty", "run_id",
            "rep_n", "rep_spread", "rep_sd", "env_lo", "env_hi", "joint",
            *[f"clk_{arm}" for arm in PWR.ARMS], "low_cells"]


def reading(report: str | Path) -> list[str]:
    """One PAIRS row's ratio half for a private_weight_reference report:

    `G seed ratio lo hi exit duty run_id`, the run's own within-run reading;
    `rep_n rep_spread rep_sd env_lo env_hi joint`, the JOINT reading over this
    seed and the earlier ones it was scored with (the report's `replicates`
    block; `none` for a run scored alone, which is also how C1 in `exit` was
    scored). The within-run interval is a bootstrap over repeats and
    understates the cross-run spread (DESIGN DECISION 14): the joint columns
    are the ones the table is scored on;
    `clk_<arm>...` each arm's median under-load clock over the ladder, off
    `treads_table`, and `low_cells`, the (arm, tread) cells whose
    `level_sides` carries LEVEL LOW. At a duty below 1 every cell is expected
    at the ceiling, so a LOW cell is the throttle signature no R3 gate reads.
    """
    p = _load(report)
    if p is None or p.get("experiment") != "private_weight_reference":
        return ["unreadable"]
    import private_weight_reference as PWR

    from moe.bench.timing import LEVEL_LOW

    iv = p.get("ratio_interval") or [None, None]
    row = [str((p.get("pinned") or {}).get("GROUP_SIZE_M", "?")),
           str(p.get("seed") if p.get("seed") is not None else "unrecorded"),
           _fmt(p.get("ratio")), _fmt(iv[0]), _fmt(iv[1]), _exit_word(p),
           str(p.get("duty", 1.0)), str(p.get("run_id") or Path(report).parent.name)]

    rep = p.get("replicates")
    if isinstance(rep, dict):
        env = rep.get("envelope") or [None, None]
        row += [str(rep.get("n", "none")), _fmt(rep.get("spread")), _fmt(rep.get("sd")),
                _fmt(env[0]), _fmt(env[1]), str(rep.get("verdict") or "none")]
    else:
        row += ["none"] * 6

    table = [r for r in (p.get("treads_table") or []) if isinstance(r, dict)]
    for arm in PWR.ARMS:
        clocks = [float(r["sm_clock_load_mhz"]) for r in table
                  if r.get("arm") == arm and r.get("sm_clock_load_mhz")]
        row.append(_fmt(statistics.median(clocks), ".0f") if clocks else "none")
    sided = [r for r in table if "level_sides" in r]
    row.append(str(sum(1 for r in sided if LEVEL_LOW in (r.get("level_sides") or [])))
               if sided else "none")
    return row


def pairs(reports: list[str]) -> list[str]:
    """The reports a later seed may be scored WITH: this arm's, measured, with a
    ratio and an interval formed. The same rule `load_replicates` applies, so
    a seed-1 line never names a report that would refuse it."""
    out = []
    for r in reports:
        p = _load(r)
        if p is None or p.get("experiment") != "private_weight_reference":
            continue
        if p.get("synthetic"):
            continue
        iv = p.get("ratio_interval") or [None, None]
        if p.get("ratio") is None or iv[0] is None or iv[1] is None:
            continue
        out.append(r)
    return out


def eta(report: str | Path) -> list[str]:
    """`eta lo hi band exit` for a clock_elasticity report: the per-M-tile
    elasticity the arm registered and the band its interval landed in, read
    through the arm's own `band_of` so the edges are never restated.

    THE BAND IS READ ONLY OFF A PAGE WHOSE GATES STAND BEHIND IT: exit DONE or
    CLAIM_FAIL. Any other exit prints `withheld:<EXIT>` in the band field.
    `band_of` alone put CLOCK-CARRIES on session 4's G=16 cells, an interval
    above R1's admissible ceiling on a page that exits INVALID, and that word
    asserts the clock mechanism the page has just disclaimed.
    """
    p = _load(report)
    if p is None or "elasticity" not in p:
        return ["unreadable"]
    est = p["elasticity"] or {}
    value, lo, hi = est.get("value"), est.get("lo"), est.get("hi")
    word = _exit_word(p)
    if word not in BAND_QUOTABLE:
        band = f"withheld:{word}"
    elif lo is None or hi is None:
        band = "unresolved"
    else:
        import clock_elasticity as CE
        got = CE.band_of(float(lo), float(hi))
        band = got[0] if got else "STRADDLES"
    return [_fmt(value), _fmt(lo), _fmt(hi), band, word]


def _log(session: str | Path, step: str) -> Path:
    return Path(session) / "chain-logs" / f"{step}.log"


def _report(results: str | Path, log: Path, experiment: str) -> Path | None:
    rid = run_id(log, experiment)
    if not rid:
        return None
    path = Path(results) / experiment / rid / "report.json"
    return path if path.is_file() else None


def eta_for(session: str | Path, results: str | Path, g: str) -> list[str]:
    """`eta` for the G's R1 run, found through the run id its log printed, or
    the `unmeasured` row when no R1 report for that G is on disk."""
    rep = _report(results, _log(session, f"r1-g{g}"), "clock_elasticity")
    return eta(rep) if rep is not None else list(ETA_UNMEASURED)


def gate(report: str | Path, tag: str) -> str:
    """The verdict of the gate `tag` on a report, `absent` when the report
    scored no such gate, `unreadable` when there is no report to read."""
    p = _load(report)
    if p is None:
        return "unreadable"
    for g in p.get("gates") or []:
        if isinstance(g, dict) and (g.get("tag") or g.get("number")) == tag:
            return str(g.get("verdict"))
    return "absent"


def verdict(log: str) -> str:
    """What the log's RESULT lines imply, as an exit code, or NONE / UNREADABLE."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return "UNREADABLE"
    try:
        return str(exit_codes.classify_text(text))
    except exit_codes.NoGatesScored:
        return "NONE"


def _render(value) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _fixed(key: str, payloads: list[tuple[str, dict]], field: str, drop=(),
           fallback: str | None = None) -> list[str]:
    """One PAIRS-fixed row: the field's value across every report that carries
    it, MIXED when they disagree, the chain's own command line when none does."""
    seen: dict[str, list[str]] = {}
    for step, p in payloads:
        if field not in p:
            continue
        value = p[field]
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in drop}
        seen.setdefault(_render(value), []).append(step)
    if not seen:
        if fallback is None:
            return [key, "none", "no report carries it yet"]
        return [key, fallback, "the chain's command line; no report carries it"]
    if len(seen) == 1:
        (value, steps), = seen.items()
        return [key, value, f"report.json of {len(steps)} run(s)"]
    return [key, "MIXED: " + " | ".join(seen),
            "; ".join(f"{v}: {', '.join(s)}" for v, s in seen.items())]


def pairs_table(session: str | Path, results: str | Path, ladder: str, seeds: str,
                settings: dict[str, str] | None = None) -> int:
    """REWRITE PAIRS.tsv from the reports on disk, and PAIRS-fixed.tsv beside it.

    Rebuilt, never appended: the chain calls this at the end of every pass and
    before every STOP, so a step re-run on --resume cannot leave a second row
    for one run id, and an R1 that finished on a later pass is joined to every
    row of its G. One row per (G, seed) whose ratio report is on disk, in
    ladder order; the R1 columns are `eta` for the G. The coordinates every
    row shares (model, tile, pinned config, treads, repeats, duty) go in the
    sidecar with where each came from. Returns the number of rows.
    """
    session, settings = Path(session), dict(settings or {})
    rows, r3_payloads, r1_payloads = [], [], []
    for g in ladder.split():
        eta_cols = None
        for seed in seeds.split():
            step = f"r3-g{g}-s{seed}"
            rep = _report(results, _log(session, step), "private_weight_reference")
            if rep is None:
                continue
            payload = _load(rep)
            if payload is not None:
                r3_payloads.append((step, payload))
            if eta_cols is None:
                eta_cols = eta_for(session, results, g)
            rows.append(reading(rep) + eta_cols)
        r1 = _report(results, _log(session, f"r1-g{g}"), "clock_elasticity")
        if r1 is not None and (payload := _load(r1)) is not None:
            r1_payloads.append((f"r1-g{g}", payload))

    header = reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit"]
    _write_tsv(session / "PAIRS.tsv", [header, *rows])
    fixed = [
        ["key", "value", "source"],
        _fixed("model", r3_payloads, "model", fallback=settings.get("model")),
        _fixed("block_m", r3_payloads, "block_m", fallback=settings.get("block_m")),
        _fixed("r3_pinned", r3_payloads, "pinned", drop=("GROUP_SIZE_M",)),
        ["group_m", "varies: " + ladder, "the G column; the swizzle each run pinned"],
        _fixed("r3_treads", r3_payloads, "treads", fallback=settings.get("r3_treads")),
        _fixed("r3_repeats", r3_payloads, "repeats", fallback=settings.get("r3_repeats")),
        _fixed("r3_duty", r3_payloads, "duty", fallback=settings.get("r3_duty")),
        _fixed("r1_pinned", r1_payloads, "pinned", drop=("GROUP_SIZE_M",)),
        _fixed("r1_duty", r1_payloads, "duty", fallback=settings.get("r1_duty")),
        # R1's report does not record its treads or repeats: the command line is the record
        ["r1_treads", settings.get("r1_treads", "none"), "the chain's command line"],
        ["r1_repeats", settings.get("r1_repeats", "none"), "the chain's command line"],
    ]
    _write_tsv(session / "PAIRS-fixed.tsv", fixed)
    return len(rows)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    """Whole-file and atomic: a reader never sees half a table."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join("\t".join(_one_line(c) for c in r) + "\n" for r in rows))
    os.replace(tmp, path)


def calibration_dir(log: str | Path) -> str:
    """The run directory of the newest ruler calibrate_hardware.py wrote, off
    its own `[calibrate] wrote <yaml>` line, or ''."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    found = CALIBRATE_WROTE.findall(text)
    return str(Path(found[-1]).parent) if found else ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "card" and not rest:
        print("\t".join(card()))
    elif cmd == "device" and not rest:
        print(device())
    elif cmd == "run-id" and len(rest) == 2:
        print(run_id(rest[0], rest[1]))
    elif cmd == "estimate" and len(rest) == 1:
        print(estimate(rest[0]))
    elif cmd == "reading" and len(rest) == 1:
        print("\t".join(reading(rest[0])))
    elif cmd == "pairs":
        print("\n".join(pairs(rest)))
    elif cmd == "eta" and len(rest) == 1:
        print("\t".join(eta(rest[0])))
    elif cmd == "eta-for" and len(rest) == 3:
        print("\t".join(eta_for(*rest)))
    elif cmd == "gate" and len(rest) == 2:
        print(gate(rest[0], rest[1]))
    elif cmd == "verdict" and len(rest) == 1:
        print(verdict(rest[0]))
    elif cmd == "pairs-table" and len(rest) >= 4:
        settings = dict(kv.split("=", 1) for kv in rest[4:] if "=" in kv)
        n = pairs_table(rest[0], rest[1], rest[2], rest[3], settings)
        print(f"{n} row(s)")
    elif cmd == "calibration-dir" and len(rest) == 1:
        print(calibration_dir(rest[0]))
    else:
        print(f"unknown command {argv!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
