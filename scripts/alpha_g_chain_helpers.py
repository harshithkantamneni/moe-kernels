#!/usr/bin/env python
"""The Python half of scripts/alpha_g_chain.sh: every read of a report.json, a
log or a ledger that the chain needs, as one small command each, so the shell
stays a sequencer and the reading is testable off GPU.

    alpha_g_chain_helpers.py card                   -> card slug, why not (the probe's reason)
    alpha_g_chain_helpers.py device                 -> this card's UUID, as R3's DEVICE file keys it
    alpha_g_chain_helpers.py run-id LOG EXPERIMENT  -> the run id the plan page printed
    alpha_g_chain_helpers.py estimate LOG           -> seconds the arm's own --dry-run priced
    alpha_g_chain_helpers.py estimate-basis LOG     -> what that figure is made of, in words
    alpha_g_chain_helpers.py reading REPORT         -> one PAIRS row's ratio half (R3)
    alpha_g_chain_helpers.py pairs REPORT...        -> the reports that FORMED a ratio, one per line
    alpha_g_chain_helpers.py eta REPORT             -> eta, lo, hi, band, exit word (R1)
    alpha_g_chain_helpers.py eta-for SESSION RESULTS G  -> the same, for the G's R1 log
    alpha_g_chain_helpers.py gate REPORT TAG        -> that gate's verdict, `absent` or `unreadable`
    alpha_g_chain_helpers.py probe-note REPORT      -> the alignment probe's note and graph_calls
    alpha_g_chain_helpers.py verdict LOG            -> what its RESULT lines imply (2nd opinion)
    alpha_g_chain_helpers.py reads-as BAND EXIT     -> what a ratio beside that R1 word reads as
    alpha_g_chain_helpers.py pairs-table SESSION RESULTS LADDER SEEDS [key=value...]
                                                    -> rewrites PAIRS.tsv, PAIRS-by-G.tsv,
                                                       PAIRS-fixed.tsv and PAIRS-README.txt,
                                                       then prints the ruler's line and one
                                                       line per G: the bytes-rate bound and
                                                       the ceiling it used
    alpha_g_chain_helpers.py calibration-dir LOG    -> the run directory calibrate wrote
    alpha_g_chain_helpers.py ncu-locate GLOBS       -> the ncu the counter probe runs, where it
                                                       was found, and every candidate
    alpha_g_chain_helpers.py counters SESSION LOG RC CAP SECS BINARY WHERE CANDIDATES GLOBS
                                                    -> writes $SESSION/COUNTERS off the probe's
                                                       COUNTERS.json; prints the ledger note

Every command prints tab-separated fields on ONE line (or one path per line for
`pairs`, and `pairs-table`'s count line then its per-G lines) and exits 0; a
report that cannot be read prints `unreadable` in the first field rather than
crashing the chain.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
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
#: R3's plan names the alignment probe's seconds INSIDE its GPU figure, and
#: below full duty its wall line leaves them OUT ("the probe is timed at full
#: duty"): the price of a run below full duty is the wall line plus these.
PROBE_SECONDS = re.compile(r"includes the alignment probe's (\d+) s")
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

#: WHAT A RATIO CAN BE READ AS, off its G's R1 word: the rule H200 session 5's
#: findings support (2026-09-23, sections 3.6, 4.1 and 4.2). Keyed by the
#: names of clock_elasticity's BANDS; a test holds the keys to them.
#:   RAW-STANDS     the per-tile cost is a traffic quantity, and by the secant
#:                  caveat that carries up to R3's operating point, so
#:                  slope(shared) / slope(private) is a ratio of traffic: a
#:                  re-read fraction.
#:   CLOCK-CARRIES  session 5 at G >= 4: the shared arm sits on a per-tile
#:                  floor that scales with the SM clock (an overlap of traffic
#:                  and an on-chip floor fits R1's cells at 1.06% rms, the
#:                  additive form 3.9-5.0%), and any alpha in [0, 0.60] fits
#:                  equally. The bytes-rate bound still proves real reuse
#:                  there, so the ratio is neither a traffic fraction nor a pure
#:                  time ratio: a blend, and not alpha.
#: Every other word (STRADDLES, UNREGISTERED-GAP, withheld:<EXIT>, unmeasured,
#: unresolved) licenses no reading: READS_AS_UNRESOLVED.
READS_AS = {
    "RAW-STANDS": "re-read fraction",
    "CLOCK-CARRIES": "blend (traffic and a clock-scaled on-chip floor): not alpha",
}
READS_AS_UNRESOLVED = "unresolved"

#: The ncu the counter probe finds on PATH is named by this where-word; one
#: found by a glob is named by the glob.
ON_PATH = "PATH"
#: The file the counter probe's payload is written to (dram_counter_route.py
#: --probe --out) and the text beside it, both in the session directory.
COUNTERS_JSON = "COUNTERS.json"
COUNTERS_TEXT = "COUNTERS"


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


def _estimate_terms(log: str | Path) -> list[tuple[int, str]]:
    """The seconds an arm's own --dry-run priced, term by term, off its plan
    page, or [] when it priced nothing. R3 below full duty is TWO terms: the
    ladder's wall line at the duty and the alignment probe's seconds, which
    that line leaves out because the probe is timed at full duty (the driver
    books the same arm as their sum)."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return []
    for k, pattern in enumerate(ESTIMATE_LINES):
        m = pattern.search(text)
        if not m:
            continue
        secs = int(m.group(1))
        if k == 0:
            probe = PROBE_SECONDS.search(text)
            terms = [(secs, "the ladder's wall at the plan's duty")]
            if probe:
                terms.append((int(probe.group(1)),
                              "the alignment probe, timed at full duty on top of it"))
            return terms
        if k == 1:
            return [(secs, "the plan's GPU figure at full duty, the alignment probe inside it")]
        return [(secs, "the plan's wall figure")]
    return []


def estimate(log: str | Path) -> str:
    """Seconds the arm's own --dry-run priced, off its plan page; '' if none.
    The sum of `_estimate_terms`."""
    terms = _estimate_terms(log)
    return str(sum(s for s, _ in terms)) if terms else ""


def estimate_basis(log: str | Path) -> str:
    """What `estimate` is made of, in words: `581 s the ladder's wall ... +
    9 s the alignment probe ...`; '' if the plan priced nothing."""
    return " + ".join(f"{s} s {what}" for s, what in _estimate_terms(log))


def reading_header() -> list[str]:
    """The ratio half of PAIRS.tsv's header, the arm names read from the arm."""
    import private_weight_reference as PWR
    return ["G", "seed", "ratio", "lo", "hi", "exit", "exit_scope", "duty", "run_id",
            "rep_n", "rep_spread", "rep_sd", "env_lo", "env_hi", "joint",
            *[f"clk_{arm}" for arm in PWR.ARMS], "low_cells"]


def _formed(p: dict) -> bool:
    """The report formed its own ratio and interval: the rule `run_reading`
    and `pairs` apply."""
    iv = p.get("ratio_interval") or [None, None]
    return p.get("ratio") is not None and iv[0] is not None and iv[1] is not None


def _joint_is_its_own(p: dict) -> bool:
    """Is the report's `replicates` block a reading THIS run is in: the run
    formed its own ratio (R3 builds the block from the replicates alone when
    it did not) and, where the block lists its runs by id, this run is one."""
    rep = p.get("replicates")
    if not isinstance(rep, dict) or not _formed(p):
        return False
    ids = [r.get("run_id") for r in rep.get("runs") or [] if isinstance(r, dict)]
    return not any(ids) or p.get("run_id") in ids


def reading(report: str | Path) -> list[str]:
    """One PAIRS row's ratio half for a private_weight_reference report:

    `G seed ratio lo hi exit exit_scope duty run_id`, the run's own within-run
    reading and its exit word; `exit_scope` says how C1 inside that word was
    scored: `alone` on this run's interval, `envelope` on the envelope of
    this run's and the earlier seeds' it was given through --replicate-of;
    `rep_n rep_spread rep_sd env_lo env_hi joint`, the JOINT reading over this
    seed and those earlier ones (the report's `replicates` block), FILLED ONLY
    when the block is a reading this run is in (`_joint_is_its_own`): a run
    that formed no ratio of its own gets a block built from the other runs
    alone, and its row says `none` rather than quote their envelope as its
    own. The within-run interval is a bootstrap over repeats and understates
    the cross-run spread (DESIGN DECISION 14); the per-G joint over every
    seed, whatever order they ran in, is PAIRS-by-G.tsv's (`by_g`);
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
    own = _joint_is_its_own(p)
    row = [str((p.get("pinned") or {}).get("GROUP_SIZE_M", "?")),
           str(p.get("seed") if p.get("seed") is not None else "unrecorded"),
           _fmt(p.get("ratio")), _fmt(iv[0]), _fmt(iv[1]), _exit_word(p),
           "envelope" if own else "alone",
           str(p.get("duty", 1.0)), str(p.get("run_id") or Path(report).parent.name)]

    rep = p.get("replicates")
    if own:
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
        if p.get("synthetic") or not _formed(p):
            continue
        out.append(r)
    return out


def eta(report: str | Path) -> list[str]:
    """`eta lo hi band exit` for a clock_elasticity report: the per-M-tile
    elasticity the arm registered and the band its interval landed in, read
    through the arm's own `band_of` so the edges are never restated.

    THE BAND IS READ ONLY OFF A PAGE WHOSE GATES STAND BEHIND IT: exit DONE or
    CLAIM_FAIL. Any other exit prints `withheld:<EXIT>` in the band field.
    `band_of` alone would put CLOCK-CARRIES on an interval above V7's
    admissible edge, on a page that exits INVALID, and that word asserts the
    clock mechanism the page has just disclaimed. The all-tread reading of
    session 4's G=16 cells (printed beside the claim, never gated) is such an
    interval. The claim itself, over treads 2 and deeper (D2), passes V7 on
    those cells, exits CLAIM_FAIL, and its CLOCK-CARRIES is quoted here.
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


def reads_as(band: str, eta_exit: str) -> str:
    """What the ratio beside R1's word at its G can be read as (`READS_AS`).

    A re-read fraction ONLY when R1's band is RAW-STANDS on a page whose gates
    stood behind it (exit DONE or CLAIM_FAIL, `BAND_QUOTABLE`); a blend, not
    alpha, on CLOCK-CARRIES; `unresolved` for every other word. `eta` already
    withholds a word from a page its gates refused, and the exit is asked
    again here so a band column edited by hand cannot license a reading."""
    if eta_exit not in BAND_QUOTABLE:
        return READS_AS_UNRESOLVED
    return READS_AS.get(band, READS_AS_UNRESOLVED)


# --------------------------------------------------------------------------
# THE BYTES-RATE BOUND: the one bound on alpha that needs no private arm
# --------------------------------------------------------------------------

def bytes_rate_bound(top_ms: float, top_tread: int, expert_set_bytes: float,
                     ceiling_gbps: float) -> float | None:
    """alpha <= (t x C / W - 1) / (n - 1), H200 session 5's findings, 3.6.

    At its top tread n the shared arm reads the expert set W once (every
    expert's weights come from DRAM at least once) and alpha x W for each
    of the n - 1 later M-tiles, the study's byte
    model R(n) = W (1 + alpha (n - 1)) (dram_counter_route.py). The flushed
    L2 is a small fraction of W, so the first read is compulsory. Those bytes
    cannot arrive faster than the ceiling C, so in the tread's time t they
    are at most t x C, which bounds alpha from above with no private arm, no
    fitted slope and no assumed rate beyond C. Activation and output bytes
    are left out, which can only loosen it. At 1 or above it excludes
    nothing: a full re-read per M-tile fits in the time.

    It is written with the compulsory first read because that is the form
    that reproduces the findings' figures (0.886 at the pin rate, 0.842 at
    read_stream, 0.789 at triad, off the G >= 4 seed means); t x C / (n x W),
    without it, is the looser bound on bytes per M-tile. None when the
    inputs cannot carry it (one tread, or a zero or negative term)."""
    if top_tread < 2 or top_ms <= 0 or expert_set_bytes <= 0 or ceiling_gbps <= 0:
        return None
    reads_in_time = top_ms * 1e-3 * ceiling_gbps * 1e9 / expert_set_bytes
    return (reads_in_time - 1.0) / (top_tread - 1)


def shared_top(payload: dict) -> tuple[int, float] | None:
    """(top tread, ms) off the shared arm's ladder in an R3 report: the
    deepest point the report's own ladder carries, which is the per-tread
    median `ladder_for` wrote. None when the report carries no such ladder."""
    import private_weight_reference as PWR
    points = (((payload.get("ladders") or {}).get(PWR.SHARED) or {}).get("points")) or []
    try:
        n, ms = max((int(p[0]), float(p[1])) for p in points)
    except (TypeError, ValueError, IndexError):
        return None
    return n, ms


def expert_set_bytes(payload: dict) -> int | None:
    """W, the bytes of one copy of the layer's expert set, off the report's
    own memory plan (`per_copy_bytes`: every expert's w1 and w2, once)."""
    try:
        value = int((payload.get("memory_plan") or {})["per_copy_bytes"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def load_ruler(path: str | Path) -> dict:
    """The two ceilings the bound is quoted at, off a calibrate_hardware.py
    yaml, never typed: the read ceiling (the MATCHED_CEILING pattern,
    read_stream, unless the ruler disowned it) and the memory bus pin rate
    (`observed.pin_rate_gbps`). read_reduce is NOT a fallback: calibrate calls
    it a LOWER bound on the read rate, and a ceiling set too low makes this
    bound too tight. Also the ruler's named bandwidth, which the check against
    the reports reads. Raises ValueError on a file that is not a ruler."""
    import yaml

    from moe.bench import calibrate as CAL
    try:
        doc = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: {exc.__class__.__name__}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a mapping")
    detail = doc.get("detail") or {}
    patterns = {p.get("pattern"): p for p in detail.get("bandwidth_patterns") or []
                if isinstance(p, dict)}
    read, read_why = patterns.get(CAL.MATCHED_CEILING), ""
    if read is None:
        read_gbps, read_why = None, f"no {CAL.MATCHED_CEILING} pattern on this ruler"
    elif CAL.DISOWNED in str(read.get("note") or ""):
        read_gbps, read_why = None, f"this ruler disowns its {CAL.MATCHED_CEILING}: {read['note']}"
    else:
        read_gbps = float(read["gbps"])
    pin = (doc.get("observed") or {}).get("pin_rate_gbps")
    named = detail.get("achieved_bandwidth_gbps")
    if named is None and (doc.get("memory") or {}).get("bandwidth_tb_s"):
        named = float(doc["memory"]["bandwidth_tb_s"]) * 1000.0
    return {"path": str(path), "read_pattern": CAL.MATCHED_CEILING, "read_gbps": read_gbps,
            "read_why": read_why, "pin_gbps": float(pin) if pin else None,
            "named_gbps": float(named) if named else None,
            "named_pattern": str(detail.get("ceiling_pattern") or "its named"),
            "checked_on": str(doc.get("checked_on") or "undated"),
            "commit": str(doc.get("measured_commit") or "")[:7] or "no commit recorded"}


def session_ruler(session: str | Path, card: str | None,
                  explicit: str | None = None) -> tuple[Path | None, str]:
    """WHICH RULER THIS SESSION MEASURED, and how it was found: `ruler=` when
    given; else the yaml calibrate wrote in this session (the last
    `[calibrate] wrote ...yaml` line of $SESSION/logs/calibrate.log that is
    still on disk, which on the laptop after exfil it is not); else the
    tracked moe/bench/hardware/measured_<card>.yaml, which calibrate --publish
    overwrote on the pod and the results commit carries. `ruler_for` then
    holds it to the bandwidth the reports were scored against."""
    if explicit:
        return Path(explicit), "given as ruler="
    log = Path(session) / "logs" / "calibrate.log"
    try:
        wrote = CALIBRATE_WROTE.findall(log.read_text(errors="replace"))
    except OSError:
        wrote = []
    on_disk = [w for w in wrote if Path(w).is_file()]
    if on_disk:
        return Path(on_disk[-1]), f"the yaml calibrate wrote in this session ({log})"
    if card:
        tracked = ROOT / "moe" / "bench" / "hardware" / f"measured_{card}.yaml"
        if tracked.is_file():
            return tracked, (f"the tracked ruler for {card}"
                             + (f" ({log} names none on this disk)" if wrote else ""))
    return None, (f"no ruler: {log} names no yaml on this disk and there is no tracked "
                  f"measured_{card or '<card>'}.yaml; pass ruler=<yaml> to pairs-table")


def ruler_for(session: str | Path, payloads: list[dict],
              explicit: str | None = None) -> tuple[dict | None, str]:
    """The session's ruler, CHECKED: its named bandwidth must be the one every
    report that records one was scored against (`bandwidth_gbps`), or the
    bound is withheld. calibrate --publish overwrites ONE file per card, and a
    ruler from another calibration beside these reports is the failure
    moe.bench.calibrate.CalibrationStamp records. Returns (ruler or None, why)."""
    cards = {str(p["card"]) for p in payloads if p.get("card")}
    path, how = session_ruler(session, next(iter(cards)) if len(cards) == 1 else None, explicit)
    if path is None:
        return None, how
    try:
        ruler = load_ruler(path)
    except ValueError as exc:
        return None, _one_line(f"{how}: {path} is not a ruler: {exc}")
    scored = {float(p["bandwidth_gbps"]) for p in payloads if p.get("bandwidth_gbps")}
    if not scored:
        return ruler, f"{how}; unchecked: no report records the bandwidth it was scored against"
    named = ruler["named_gbps"]
    if named is None or any(not math.isclose(named, b, rel_tol=1e-9) for b in scored):
        return None, _one_line(
            f"{how}: {path} (named bandwidth {_fmt(named, '.4f')} GB/s) is not the ruler these "
            f"reports were scored against ({', '.join(f'{b:.4f}' for b in sorted(scored))} GB/s); "
            "pass ruler=<the session's calibrate yaml> to pairs-table")
    return ruler, (f"{how}; its {ruler['named_pattern']} {named:.1f} GB/s is the bandwidth "
                   f"every report here was scored against")


def g_bound(payloads: list[dict], ruler: dict | None) -> tuple[list[str], dict]:
    """One PAIRS-by-G row's bound columns, `top_tread shared_top_ms
    bound_read_stream bound_pin_rate`, and what the console line needs. The
    time is the MEAN of the shared arm's top-tread point over the G's runs
    (the reading the findings quote); W is theirs, and must agree."""
    tops = [t for t in (shared_top(p) for p in payloads) if t is not None]
    sets = {w for w in (expert_set_bytes(p) for p in payloads) if w is not None}
    info = {"n": len(tops), "why": ""}
    if not tops or len({n for n, _ in tops}) != 1 or len(sets) != 1:
        info["why"] = ("no shared ladder or memory plan on any run" if not tops or not sets
                       else "the runs disagree on the top tread or the expert set")
        return ["none"] * 4, info
    n_top, w = tops[0][0], sets.pop()
    ms = statistics.mean(t for _, t in tops)
    info.update(top_tread=n_top, ms=ms, w=w)
    bounds = []
    for key in ("read_gbps", "pin_gbps"):
        c = (ruler or {}).get(key)
        b = bytes_rate_bound(ms, n_top, w, c) if c else None
        info[key] = b
        bounds.append(_fmt(b))
    return [str(n_top), _fmt(ms), *bounds], info


def ruler_line(ruler: dict | None, ruler_why: str) -> str:
    """The console's line naming the ruler every G's bound below is read at,
    or why there is none; said once, above the per-G lines."""
    if ruler is None:
        return f"bytes-rate bound: NO RULER, so no G has one: {ruler_why}"
    return f"bytes-rate bound, at the ceilings of {ruler['path']}: {ruler_why}"


def bound_line(g: str, info: dict, ruler: dict | None, label: str) -> str:
    """The console's line for one G: the bound with the ceiling it used, or
    why there is none, and what the ratio reads as."""
    head = f"G={g:<3} "
    if ruler is None:
        return f"{head}no bytes-rate bound (no ruler, above); reads as: {label}"
    if info.get("ms") is None:
        return f"{head}no bytes-rate bound: {info['why']}; reads as: {label}"
    parts = []
    for key, name, missing in (
            ("read_gbps", ruler["read_pattern"], ruler["read_why"]),
            ("pin_gbps", "the pin rate", "no observed.pin_rate_gbps on this ruler")):
        b, c = info.get(key), ruler.get(key)
        parts.append(f"<= {b:.4f} at {name} {c:.1f} GB/s" if b is not None
                     else f"no bound at {name} ({missing})")
    vacuous = [b for b in (info.get("read_gbps"), info.get("pin_gbps")) if b is not None and b >= 1]
    tail = ("; 1 or above: a full re-read per M-tile fits in that time, so the bound excludes "
            "nothing there" if vacuous else "")
    seeds = f"the mean of {info['n']} run(s)"
    return (f"{head}alpha {', '.join(parts)} (bytes-rate bound: the shared arm's tread "
            f"{info['top_tread']} at {info['ms']:.4f} ms, {seeds}, over a "
            f"{info['w'] / 1e9:.4f} GB expert set){tail}; reads as: {label}")


#: PAIRS-by-G.tsv's header: the R3 half off `by_g`, then R1's columns, what the
#: ratio reads as, and the bytes-rate bound.
BY_G_HEADER = ["G", "n", "seeds", "mean", "sd", "env_lo", "env_hi", "joint",
               "invalid_in_envelope", "eta", "eta_lo", "eta_hi", "band", "eta_exit", "reads_as",
               "top_tread", "shared_top_ms", "bound_read_stream", "bound_pin_rate", "note"]


def by_g(g: str, reports: list[str]) -> tuple[list[str], str]:
    """One PAIRS-by-G row's R3 half, and its note: EVERY report of this G
    that formed a ratio, whatever order its seeds ran in, read together by
    R3's own cross-run machinery the way `--read RUN --replicate-of ...`
    reads them (`run_reading` of the first, `load_replicates` of the rest
    against its design, `cross_run` over all), so the envelope, the sd and
    the joint verdict are R3's and are written nowhere else.

    `n seeds mean sd env_lo env_hi joint invalid_in_envelope`: `sd` is the
    points' (R3 forms one from three); `joint` is C1's rule on the envelope;
    `invalid_in_envelope` names the seeds inside it whose own page exited
    INVALID, which `load_replicates` admits by design (their spread is
    information, their ratio is not quotable alone). A refusal by
    `load_replicates` (two duties, say) is `REFUSED` with its reason.
    """
    import private_weight_reference as PWR
    formed = pairs(reports)
    if not formed:
        return [g, "0"] + ["none"] * 7, "no report of this G formed a ratio yet"
    payload = _load(formed[0]) or {}
    seeds = " ".join(str(s) if (s := (_load(r) or {}).get("seed")) is not None
                     else "unrecorded" for r in formed)
    try:
        this = PWR.run_reading(payload, Path(formed[0]))
        reps = PWR.load_replicates(
            formed[1:], card_known=True, this=this,
            design={k: payload.get(k, PWR.DESIGN_KEY_DEFAULTS.get(k)) for k in PWR.DESIGN_KEYS})
    except PWR.PrivateWeightRefusal as exc:
        return ([g, str(len(formed)), seeds] + ["none"] * 4 + ["REFUSED", "none"],
                _one_line(f"load_replicates refused: {exc}"))
    cross = PWR.cross_run([this, *reps])
    lo, hi = cross.envelope
    invalid = [str(r.seed if r.seed is not None else r.name) for r in cross.readings
               if r.exit_code == exit_codes.INVALID]
    note = ("one run: the envelope is its own interval" if len(cross.readings) == 1
            else "an sd needs three points" if cross.sd is None else "")
    return ([g, str(len(cross.readings)), seeds, _fmt(statistics.mean(cross.points)),
             _fmt(cross.sd), _fmt(lo), _fmt(hi), cross.verdict,
             ("seed " + ", seed ".join(invalid)) if invalid else "none"], note or "none")


def pairs_readme(r1_duty: str = "none", r3_duty: str = "none") -> str:
    """PAIRS-README.txt: what every column of the three tables means, the
    interval widths read off the arms (R3's INTERVAL_PCT and ALPHA_BAND), and
    the duty states R1's word is a secant across and R3's duty, as
    PAIRS-fixed.tsv's r1_duty and r3_duty rows read them (pairs_table passes
    those values). The sentence on the chain's defaults names R1_DUTY's and
    R3_DUTY's defaults in scripts/alpha_g_chain.sh; a test holds it to them."""
    import private_weight_reference as PWR
    pct = f"{PWR.INTERVAL_PCT:.0f}%"
    band = f"[{PWR.ALPHA_BAND[0]}, {PWR.ALPHA_BAND[1]})"
    reread, blend = READS_AS["RAW-STANDS"], READS_AS["CLOCK-CARRIES"]
    unresolved = READS_AS_UNRESOLVED
    return f"""\
THE alpha(G) TABLES, rebuilt from the reports on disk by
scripts/alpha_g_chain_helpers.py pairs-table at the end of every chain pass and
before every STOP, and again on the laptop after exfil by the same command.

PAIRS.tsv, one row per ratio run (private_weight_reference), G then seed:
  G seed duty run_id   the swizzle the run pinned (GROUP_SIZE_M), its seed, its
                       duty and its run id.
  ratio lo hi          the run's OWN reading, slope(shared) / slope(private), and
                       its {pct} percentile bootstrap over repeats WITHIN the run
                       (R3's INTERVAL_PCT). It understates the run-to-run spread.
                       Both slopes over the window the report records
                       (claim_min_tread): treads {PWR.CLAIM_MIN_TREAD} and deeper
                       since 2026-09-23, every tread on a report before it.
                       Reports of two windows are refused, not pooled.
  exit                 the run's own exit word: classify over its page's gates.
  exit_scope           how C1 inside that word was scored: `alone` on this run's
                       interval; `envelope` on the envelope of this run's interval
                       and those of the earlier seeds it was given through
                       --replicate-of. Seed 0's exit and seed 1's can differ by
                       scope, not by result.
  rep_n rep_spread rep_sd env_lo env_hi joint
                       the joint reading on that run's page, over itself and those
                       earlier seeds: n, the spread and sd of the points (an sd
                       needs three), the envelope, and `joint`, C1's verdict on the
                       envelope against R3's refit band ALPHA_BAND {band}. `joint`
                       is NOT a quotability flag: NO-REUSE, expected at G=1, reads
                       FAIL. `none` on a run scored alone, and on a run whose page
                       formed no ratio of its own (its page's replicates block is
                       then the other runs' reading, not one this run is in).
  clk_<arm> low_cells  each arm's median under-load clock (MHz) over the ladder, and
                       the (arm, tread) cells whose level record reads LEVEL LOW.
  eta eta_lo eta_hi band eta_exit
                       R1 (clock_elasticity) at this G: the per-M-tile elasticity
                       over treads 2 and deeper, its 95% percentile bootstrap
                       (clock_elasticity.fit's 2.5th and 97.5th percentiles), the
                       regime word off that interval, and R1's own exit word. The
                       word is withheld (`withheld:<EXIT>`) from a page whose gates
                       did not stand behind it. It is a SECANT across R1's duty
                       states, not a local reading at R3's duty. This session's R1
                       states: {r1_duty}; R3's duty: {r3_duty} (PAIRS-fixed.tsv's
                       r1_duty and r3_duty). At the chain's defaults, R1_DUTY
                       1.0 0.5 0.25 and R3_DUTY 0.25, the secant runs from the
                       capped clock at 1.0 to the clocks at 0.5 and 0.25 (on
                       session 5's H200 only 1.0 held the 700 W cap), and R3's
                       0.25 is the top of that range.
  reads_as             what the ratio can be read as, off R1's word at this G and
                       the rule H200 session 5's findings support: `{reread}`
                       ONLY when the band is RAW-STANDS on a page whose gates
                       stood behind it; `{blend}` on CLOCK-CARRIES (session 5 at
                       G >= 4: the shared arm sits on a per-tile floor that
                       scales with the SM clock, any alpha in [0, 0.60] fits it
                       equally, and the bytes-rate bound below still proves real
                       reuse); `{unresolved}` on STRADDLES, UNREGISTERED-GAP,
                       withheld:<EXIT> and unmeasured. It is R1's reading at the
                       G; whether this run's ratio is quotable at all is `exit`.

PAIRS-by-G.tsv, one row per G, THE PER-G VALUE: every seed of that G whose report
formed a ratio, whatever order the seeds ran in, read together by R3's own
cross-run machinery (what `--read RUN --replicate-of ...` prints):
  n seeds              how many runs, and their seeds.
  mean sd              the points' mean, and their sd (three points or more).
  env_lo env_hi joint  the envelope of the runs' {pct} intervals and C1's verdict on
                       it, the same rule and band as `joint` above.
  invalid_in_envelope  the seeds inside the envelope whose own page exited INVALID:
                       R3 admits them by design, their spread is information and
                       their ratio is not quotable alone.
  eta .. eta_exit      R1 at this G, as above.
  reads_as             as above.
  top_tread shared_top_ms bound_read_stream bound_pin_rate
                       THE BYTES-RATE BOUND, the one bound on alpha that needs no
                       private arm: alpha <= (t x C / W - 1) / (n - 1), with t the
                       shared arm's time at its top tread n off each report's own
                       ladder (shared_top_ms, the mean over these runs), W the
                       expert set (PAIRS-fixed.tsv's expert_set_bytes) and C a
                       ceiling off the ruler the session measured
                       (PAIRS-fixed.tsv's read_ceiling_gbps and pin_rate_gbps):
                       the shared arm reads W once and alpha x W for each later
                       M-tile, and no faster than C. At read_stream, the ruler's
                       read ceiling, and at the pin rate, the bus's hard one. 1 or
                       above excludes nothing (a full re-read fits in the time);
                       `none` when the ruler or the ladder is missing, and
                       PAIRS-fixed.tsv's ruler row says why.
  note                 why a G has no joint (no report formed a ratio, or
                       load_replicates refused the set, e.g. two duties).

PAIRS-fixed.tsv: the coordinates every row shares (model, tile, pinned config,
treads, repeats, duty), their value and where each was read; MIXED when the
reports disagree. Also the bound's inputs: expert_set_bytes (each report's
memory_plan.per_copy_bytes), the ruler (which yaml, and the check that its named
bandwidth is the one the reports were scored against), read_ceiling_gbps and
pin_rate_gbps. On the laptop the session's calibrate yaml is not on disk, so the
tracked ruler is read and checked; `ruler=<yaml>` on pairs-table names another.
"""


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


def probe_note(report: str | Path) -> str:
    """The alignment probe's own record off an R3 report, for the V8 STOP:
    its note (a refused capture is named there), the `graph_calls` its cells
    were timed with (0 is eager) and how many cells the instrument called
    host-bound. '' when the report carries no probe."""
    p = _load(report)
    probe = (p or {}).get("align_probe")
    if not isinstance(probe, dict):
        return ""
    cells = [c for c in probe.get("cells") or [] if isinstance(c, dict)]
    parts = [f"note: {probe.get('note') or 'none'}"]
    if cells:
        calls = sorted({int(c.get("graph_calls") or 0) for c in cells})
        judged = [c for c in cells if c.get("host_bound") is not None]
        parts.append("graph_calls " + ", ".join(map(str, calls))
                     + (" (0 is eager)" if 0 in calls else ""))
        parts.append(f"host-bound {sum(1 for c in judged if c['host_bound'])} "
                     f"of {len(judged)} judged cells")
    return _one_line("; ".join(parts))


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
    """REWRITE PAIRS.tsv, PAIRS-by-G.tsv and PAIRS-fixed.tsv from the reports
    on disk, and PAIRS-README.txt, the legend, beside them.

    Rebuilt, never appended: the chain calls this at the end of every pass and
    before every STOP, so a step re-run on --resume cannot leave a second row
    for one run id, and an R1 that finished on a later pass is joined to every
    row of its G. PAIRS.tsv: one row per (G, seed) whose ratio report is on
    disk, in ladder order; the R1 columns are `eta` for the G. PAIRS-by-G.tsv:
    one row per G of the ladder, every seed read together (`by_g`). The
    coordinates every row shares (model, tile, pinned config, treads,
    repeats, duty) go in the sidecar with where each came from, and the legend
    names the sidecar's r1_duty and r3_duty. Every row of both tables says
    what its ratio reads as (`reads_as`), and every PAIRS-by-G row carries the
    bytes-rate bound (`g_bound`) at the ceilings of the ruler the session
    measured (`ruler_for`; `settings["ruler"]` names another). Returns the
    number of PAIRS.tsv rows; `pairs_table_lines` returns the console's lines
    beside it.
    """
    return pairs_table_lines(session, results, ladder, seeds, settings)[0]


def pairs_table_lines(session: str | Path, results: str | Path, ladder: str, seeds: str,
                      settings: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """`pairs_table`, and its console lines: the ruler the bounds are read at
    (or why there is none), then one line per G, its bytes-rate bound with the
    ceiling it used, or why there is none, and what its ratio reads as."""
    session, settings = Path(session), dict(settings or {})
    rows, by_g_rows, r3_payloads, r1_payloads = [], [], [], []
    per_g = []
    for g in ladder.split():
        eta_cols = eta_for(session, results, g)
        label = reads_as(*eta_cols[3:5]) if len(eta_cols) == 5 else READS_AS_UNRESOLVED
        found = []
        for seed in seeds.split():
            step = f"r3-g{g}-s{seed}"
            rep = _report(results, _log(session, step), "private_weight_reference")
            if rep is None:
                continue
            found.append(str(rep))
            payload = _load(rep)
            if payload is not None:
                r3_payloads.append((step, payload))
            rows.append(reading(rep) + eta_cols + [label])
        r3_half, note = by_g(g, found)
        per_g.append((g, r3_half, eta_cols, label, note,
                      [p for r in pairs(found) if (p := _load(r)) is not None]))
        r1 = _report(results, _log(session, f"r1-g{g}"), "clock_elasticity")
        if r1 is not None and (payload := _load(r1)) is not None:
            r1_payloads.append((f"r1-g{g}", payload))

    ruler, ruler_why = ruler_for(session, [p for _, p in r3_payloads], settings.get("ruler"))
    lines = [ruler_line(ruler, ruler_why)]
    for g, r3_half, eta_cols, label, note, formed in per_g:
        bound_cols, info = g_bound(formed, ruler)
        by_g_rows.append(r3_half + eta_cols + [label] + bound_cols + [note])
        lines.append(bound_line(g, info, ruler, label))

    header = reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit", "reads_as"]
    _write_tsv(session / "PAIRS.tsv", [header, *rows])
    _write_tsv(session / "PAIRS-by-G.tsv", [BY_G_HEADER, *by_g_rows])
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
        # the bytes-rate bound's inputs: W off every report, the ceilings off the ruler
        _fixed("expert_set_bytes", [(s, {"w": w}) for s, p in r3_payloads
                                    if (w := expert_set_bytes(p)) is not None], "w"),
        ["ruler", ruler["path"] if ruler else "none", ruler_why],
        ["read_ceiling_gbps",
         f"{ruler['read_gbps']:.1f}" if ruler and ruler["read_gbps"] else "none",
         (f"the ruler's {ruler['read_pattern']} pattern, measured {ruler['checked_on']} at "
          f"{ruler['commit']}" if ruler and ruler["read_gbps"]
          else (ruler["read_why"] if ruler else "no ruler"))],
        ["pin_rate_gbps", f"{ruler['pin_gbps']:.1f}" if ruler and ruler["pin_gbps"] else "none",
         ("the ruler's observed.pin_rate_gbps (memory clock x 2 x bus width)"
          if ruler and ruler["pin_gbps"] else "not on the ruler" if ruler else "no ruler")],
    ]
    _write_tsv(session / "PAIRS-fixed.tsv", fixed)
    # the legend names the duties the fixed rows read, without the flag a
    # value off the chain's command line carries
    duty = {r[0]: r[1].removeprefix("--duty ") for r in fixed[1:]}
    tmp = session / "PAIRS-README.txt.tmp"
    tmp.write_text(pairs_readme(duty["r1_duty"], duty["r3_duty"]))
    os.replace(tmp, session / "PAIRS-README.txt")
    return len(rows), lines


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


# --------------------------------------------------------------------------
# THE COUNTER PROBE: can this pod read a DRAM counter (informational)
# --------------------------------------------------------------------------

def ncu_locate(globs: list[str], path_env: str | None = None) -> list[tuple[str, str]]:
    """Every ncu the chain's counter probe would consider, in the order it
    would run one: the one a bare `ncu` runs (PATH), then each glob's matches,
    newest name first (reverse lexical: /usr/local/cuda/bin/ncu, the image's
    default symlink, before /usr/local/cuda-13.0/bin/ncu, and 2025.3 before
    2024.1 under /opt/nvidia/nsight-compute). Each is (path, where): `PATH`
    or the glob that found it; a path found twice is listed once.

    dram_counter_route.py's probe asks PATH alone, and session 4's image read
    "no ncu on PATH" there: an ncu under the CUDA toolkit's bin or Nsight
    Compute's own directory, off PATH, was never looked for. The chain runs
    that same probe with the first match's directory put first on PATH, so
    the probe's own code decides what it can read, and nothing is restated."""
    found: list[tuple[str, str]] = []
    on_path = shutil.which("ncu", path=path_env)
    if on_path:
        found.append((on_path, ON_PATH))
    for pattern in globs:
        for cand in sorted(glob.glob(pattern), reverse=True):
            if (os.path.isfile(cand) and os.access(cand, os.X_OK)
                    and cand not in {p for p, _ in found}):
                found.append((cand, pattern))
    return found


def _counter_error_line(payload: dict, log_text: str) -> str:
    """The refusal ncu printed, verbatim: the first line carrying
    ERR_NVGPUCTRPERM in the probe's own captured output, then in its console;
    the probe's cause when neither carries one."""
    ncu = payload.get("ncu") or {}
    for text in (str(ncu.get("output_head") or ""), log_text):
        for line in text.splitlines():
            if "ERR_NVGPUCTRPERM" in line:
                return line.strip()
    return str(ncu.get("cause") or ncu.get("why") or "no error recorded")


def _caps_text(payload: dict) -> str:
    """The two capabilities either of which opens the counter gate, and the
    host's module flag, as the probe read them."""
    caps, flag = payload.get("capabilities") or {}, payload.get("module_flag") or {}
    if caps.get("available"):
        c = (f"CAP_PERFMON {'set' if caps.get('perfmon') else 'clear'}, CAP_SYS_ADMIN "
             f"{'set' if caps.get('sys_admin') else 'clear'} (CapEff "
             f"{caps.get('cap_eff_field', caps.get('cap_eff'))})")
    else:
        c = f"capabilities unread ({caps.get('why', 'no record')})"
    if flag.get("available"):
        m = f"RestrictProfilingToAdminUsers={flag.get('restrict')}"
    else:
        m = f"module flag unread ({flag.get('why', 'no record')})"
    return f"{c}; {m}"


def counters(session: str | Path, log: str | Path, rc: str | int, cap: str | int,
             secs: str | int, binary: str, where: str, candidates: str, globs: str) -> str:
    """THE COUNTER PROBE'S VERDICT, off dram_counter_route.py --probe
    --family r3-arms's own payload ($SESSION/COUNTERS.json) and log, written
    to $SESSION/COUNTERS and returned as the ledger's one-line note.
    INFORMATIONAL: nothing gates on it, and the chain never latches it. The
    first word is the verdict:

      OPEN      a kernel launched under ncu and every STRICT metric of the
                r3-arms family came back (the probe's OPEN, P1 PASS): the R3
                counter run, findings section 7's, can happen on this pod.
      BLOCKED   ncu ran, a kernel launched, and the counter read was refused
                (the probe's BLOCKED, ERR_NVGPUCTRPERM); the exact line is
                quoted, with the two capabilities and the module flag.
      ABSENT    no ncu on PATH and none at the searched globs.
      UNTESTED  ncu is here but the probe launched no kernel to count (no
                torch, no CUDA device, a failed launch): nothing is known.
      ERROR     the probe crashed, timed out, wrote no payload, or its exit
                code and its RESULT lines disagree (the chain's second
                opinion, `verdict`).
    `binary`, `where` and `candidates` are `ncu-locate`'s three fields
    (`none` when it found nothing); `globs` is what it searched."""
    session, rc, cap, secs = Path(session), int(rc), int(cap), int(secs)
    try:
        log_text = Path(log).read_text(errors="replace")
    except OSError:
        log_text = ""
    payload = _load(session / COUNTERS_JSON)
    implied = verdict(str(log))
    ncu = (payload or {}).get("ncu") or {}
    found = binary not in ("", "none")
    if not found:
        located = f"not on PATH and none at {globs}"
    elif where == ON_PATH:
        located = f"{binary} (on PATH)"
    else:
        located = (f"{binary} (NOT on PATH; found by {where}, and the probe ran with "
                   f"{os.path.dirname(binary)} first on PATH)")
    version = str(ncu.get("version") or "")
    last = next((ln.strip() for ln in reversed(log_text.splitlines()) if ln.strip()),
                "an empty log")
    if cap > 0 and (rc == 124 or (rc == 137 and secs >= cap)):
        word, detail = "ERROR", (f"TIMED OUT after {secs} s against a cap of {cap} s (exit "
                                 f"{rc}); the next pass asks again")
    elif payload is None:
        word, detail = "ERROR", (f"the probe wrote no {COUNTERS_JSON} (exit {rc}); its log "
                                 f"ends: {last}")
    elif not (implied == str(rc) or (implied == "NONE" and rc == exit_codes.REFUSED)):
        word, detail = "ERROR", (f"DEFECT: the probe exited {rc} and its RESULT lines imply "
                                 f"{implied}; its page reads {payload.get('verdict')}")
    elif payload.get("verdict") == "OPEN":
        word, detail = "OPEN", str(ncu.get("cause") or "a counter came back")
    elif payload.get("verdict") == "BLOCKED":
        word, detail = "BLOCKED", _counter_error_line(payload, log_text)
    elif payload.get("verdict") == "REFUSE" and not ncu.get("present") and found:
        word, detail = "ERROR", (f"DEFECT: {binary} was found and its directory put first on "
                                 f"PATH, and the probe still read {ncu.get('why', 'no ncu')}")
    elif payload.get("verdict") == "REFUSE" and not ncu.get("present"):
        word, detail = "ABSENT", "no ncu where the chain looks"
    elif payload.get("verdict") == "REFUSE":
        word, detail = "UNTESTED", str(ncu.get("cause") or "the probe launched no kernel")
    else:
        word, detail = "ERROR", f"the probe's page reads verdict {payload.get('verdict')!r}"
    caps = _caps_text(payload or {})
    off_path = found and where != ON_PATH
    lines = [
        f"THE COUNTER PROBE, {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        " (informational: it gates nothing, and the chain asks again on every pass)",
        f"verdict     {word}",
        f"detail      {detail}",
        f"ncu         {located}" + (f"  [{version}]" if version else ""),
        f"searched    PATH, then {globs}",
        f"candidates  {candidates if found else 'none'}",
        f"access      {caps}",
        f"probe       scripts/dram_counter_route.py --probe --family r3-arms, exit {rc} in "
        f"{secs} s; its page"
        f" {log}, its payload {session / COUNTERS_JSON}",
    ]
    if off_path:
        lines.append(f"PATH        ncu is not on PATH here: the driver's counter_plan and "
                     f"dram_counter_route.py --run look on PATH only, so run them with "
                     f"PATH={os.path.dirname(binary)}:$PATH")
    for note in (payload or {}).get("notes") or []:
        lines.append(f"probe note  {_one_line(note)}")
    lines.append("history     RunPod's record is in bash scripts/alpha_g_chain.sh --help")
    tmp = session / (COUNTERS_TEXT + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, session / COUNTERS_TEXT)
    return _one_line(f"{word}: {detail}; ncu {located}" + (f" [{version}]" if version else "")
                     + f"; {caps}; informational, gates nothing; {session / COUNTERS_TEXT}")


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
    elif cmd == "estimate-basis" and len(rest) == 1:
        print(estimate_basis(rest[0]))
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
    elif cmd == "probe-note" and len(rest) == 1:
        print(probe_note(rest[0]))
    elif cmd == "verdict" and len(rest) == 1:
        print(verdict(rest[0]))
    elif cmd == "pairs-table" and len(rest) >= 4:
        settings = dict(kv.split("=", 1) for kv in rest[4:] if "=" in kv)
        n, lines = pairs_table_lines(rest[0], rest[1], rest[2], rest[3], settings)
        print(f"{n} row(s)")
        for line in lines:
            print(line)
    elif cmd == "reads-as" and len(rest) == 2:
        print(reads_as(rest[0], rest[1]))
    elif cmd == "calibration-dir" and len(rest) == 1:
        print(calibration_dir(rest[0]))
    elif cmd == "ncu-locate" and len(rest) == 1:
        found = ncu_locate(rest[0].split())
        print("\t".join([found[0][0], found[0][1], " ".join(p for p, _ in found)] if found
                        else ["none", "none", "none"]))
    elif cmd == "counters" and len(rest) == 9:
        print(counters(*rest))
    else:
        print(f"unknown command {argv!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
