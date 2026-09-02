"""One provenance block, written by every script into every report and CSV.

A number nobody can attribute to a commit, a card and a ruler is not a
measurement, it is an anecdote. The 2026-09-02 audit found that none of the ten
gaps-session scripts wrote a git sha or a dirty flag, that the 26 published
`report.json` files carried no commit, timestamp, gpu_name, iters, warmup, timing
basis, clocks, bandwidth source or ridge source, and that the H200 s4 reports
had filenames in a run-id format that entered git three hours AFTER the session,
so no commit in the history can be named as their producer. `moe/bench/driver.py`
had the right columns all along (git_sha, git_dirty, sm_clock_*, torch/triton/
vllm versions, driver version); the scripts around it each re-invented a subset.

This module is the subset made whole and made shared. `provenance_block()`
returns one frozen `Provenance`; `.as_dict()` goes into a JSON payload,
`.as_columns()` into a CSV row, `.stamp()` puts the five keys the audit gate
checks at the top level of a report. `run_id()` names a run so that two runs
that differ in anything the operator can vary, the card first of all, never
share a directory.

WHAT "NEVER RAISES" MEANS HERE
------------------------------
`provenance_block` records what it can and NAMES what it cannot. Every field it
could not determine is None, and `missing[field]` says why: "no CUDA",
"not a git repository", "package not installed", "not supplied by caller". A
block with a None in it is complete; a block that guessed is not. So there is no
"unknown" card, no "0" sha, no default bandwidth: `resolve_bandwidth`'s silent
4374.5 GB/s fallback is exactly the shape this refuses to have. The one thing
that DOES raise is a misspelt keyword to `provenance_block`, because that is a
bug at the call site and not a fact about the machine, and swallowing it would
let `ridge_src=` vanish without trace.

A ridge or bandwidth supplied WITHOUT its source is recorded, because it is
what the run used, but its `*_source` is listed in `missing` with the reason
"supplied without a source". The audit's `ridge_source` gate then fails on that
report, which is the intended outcome: the number came from somewhere and the
script has to say where.

WHAT COUNTS AS DIRTY
--------------------
`git status --porcelain` including untracked files. The stricter reading is
deliberate: on 2026-09-01 the pod ledger recorded "0 dirty file(s)" for a
session in which `calibrate` had already rewritten a tracked yaml and every
later arm ran on a modified checkout. Untracked files are part of the tree the
code ran against, and a count of them is cheaper to explain than to omit.

THE THREE RESUME COLLISIONS `run_id` EXISTS TO PREVENT
------------------------------------------------------
All three are the same defect: a knob left out of the id, so a second run
derived the first run's directory, resumed into it, found every cell present,
measured nothing, and printed the first run's numbers under the second run's
heading. Nothing looked wrong, because every report renders its settings from
argv rather than from the cells it read.

  1. GROUP_SIZE_M. `block_m_crossing_sweep`'s id once omitted G, BN and
     num_stages. The moment `--group-m` existed, a G=16 run derived the G=1 id
     and published G=1's timings under a G=16 heading. A whole arm was lost.
  2. THE CARD. The card is not swept by any script; it is swept by the operator
     moving to another pod while the results root is a network volume that
     outlives the pod. The proof is committed:
     `results/published/2026-09-01-nvidia_h200-cross-card-s3` and
     `results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3` both
     contain `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json`, for sm_count
     132 and 108. `span_extent_separation` had the same hole in its
     `(model, num_tokens, arm)` resume key and its `restore()` ignored gpu_name,
     so a second card after a completed first run measured ZERO cells and the
     config-mismatch check compared two restored first-card rows to each other.
  3. `--iters` / `--warmup`. A `--iters 200` re-run after `--iters 50` landed in
     the same directory and printed the 50-iteration numbers under the
     200-iteration label. Timing knobs set the measured milliseconds of every
     cell; they are swept knobs, not analysis knobs.

So `run_id` REQUIRES the card (it raises `NoCard` on None or empty), puts a slug
of it at the front of the id where `ls` shows it, and hashes every swept knob in
sorted key order so the id is the same whatever order the caller named them in.
Knobs that only re-analyse an existing set of cells (`--ridge`, `--alpha`, a
band) belong OUT of the key, and the caller decides that: this function hashes
what it is given.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROVENANCE_VERSION = 1

#: Column prefix for `as_columns`, so a provenance column never collides with a
#: measurement column that happens to share its name (`iters`, `gpu_name`).
COLUMN_PREFIX = "prov_"

#: The keys the audit's publish gate checks at the top level of a report.json.
#: `stamp()` writes exactly these beside the nested block.
TOP_LEVEL_KEYS = ("git_sha", "gpu_name", "ridge_source", "bandwidth_source", "instrument")

#: Fields the caller supplies; everything else is probed from the environment.
CALLER_FIELDS = ("instrument", "ridge", "ridge_source", "bandwidth", "bandwidth_source",
                 "warmup_ms", "iters", "target_ms")

#: Distributions whose versions belong in every row. The key is the field name
#: and the value the distribution name `importlib.metadata` knows it by.
PACKAGE_FIELDS = {"torch": "torch", "triton": "triton", "vllm": "vllm", "sglang": "sglang"}


class NoCard(ValueError):
    """`run_id` was called without a card.

    The card is the knob whose omission has already cost a published arm
    (module docstring, collision 2). An id without it is not an id.
    """


class UnresolvedKnob(ValueError):
    """A swept knob's value is None.

    A None in the key would hash to the same id whether the knob was "not
    applicable" or "not yet determined". The caller resolves every knob before
    naming the run, or leaves the knob out and says so.
    """


class ProvenanceCollision(ValueError):
    """`stamp` found a top-level key already present with a different value."""


@dataclass(frozen=True)
class Provenance:
    """Everything a stranger needs to attribute a number. None means "could not
    determine", and `missing` says why for every None."""

    provenance_version: int = PROVENANCE_VERSION
    git_sha: str | None = None
    git_dirty: bool | None = None
    git_dirty_files: int | None = None
    utc: str | None = None
    hostname: str | None = None
    gpu_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    python: str | None = None
    torch: str | None = None
    triton: str | None = None
    vllm: str | None = None
    sglang: str | None = None
    instrument: str | None = None
    ridge: float | None = None
    ridge_source: str | None = None
    bandwidth: float | None = None
    bandwidth_source: str | None = None
    warmup_ms: float | None = None
    iters: int | None = None
    target_ms: float | None = None
    missing: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-serialisable, in declaration order; `missing` last."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "missing":
                continue
            out[f.name] = getattr(self, f.name)
        out["missing"] = dict(sorted(self.missing.items()))
        return out

    def as_columns(self) -> dict[str, Any]:
        """The CSV subset: every scalar field under `prov_`, plus `prov_missing`
        as one `field=reason; field=reason` string so a row still says what it
        does not know. None values stay None; csv.DictWriter writes them empty
        and the reason is in `prov_missing`."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "missing":
                continue
            out[f"{COLUMN_PREFIX}{f.name}"] = getattr(self, f.name)
        out[f"{COLUMN_PREFIX}missing"] = "; ".join(
            f"{k}={v}" for k, v in sorted(self.missing.items()))
        return out

    def stamp(self, payload: dict) -> dict:
        """Return a copy of `payload` with the block under `provenance` and the
        audit's five keys at the top level.

        Raises `ProvenanceCollision` if the payload already carries any of
        those keys with a different value, so two provenance blocks can never
        be layered over one report without someone noticing.
        """
        block = self.as_dict()
        out = dict(payload)
        for key in (*TOP_LEVEL_KEYS, "provenance"):
            mine = block if key == "provenance" else block[key]
            if key in out and out[key] != mine:
                raise ProvenanceCollision(
                    f"payload already has {key}={out[key]!r}, provenance says {mine!r}")
            out[key] = mine
        return out


# --------------------------------------------------------------------------
# probes, each a module-level function so a test can replace it
# --------------------------------------------------------------------------

def _git(root: Path | None) -> tuple[str | None, bool | None, int | None, str | None]:
    """`(sha, dirty, dirty_files, reason)`. Reason is None only when all three
    are known."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return None, None, None, "git not on PATH"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, None, None, f"git rev-parse failed: {exc.__class__.__name__}"
    if sha.returncode != 0:
        err = (sha.stderr or "").strip().splitlines()
        return None, None, None, (f"not a git repository at {root}"
                                  if not err else f"git rev-parse: {err[0]}")
    try:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                 "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return sha.stdout.strip(), None, None, (
            f"git status failed: {exc.__class__.__name__}")
    if status.returncode != 0:
        return sha.stdout.strip(), None, None, "git status returned non-zero"
    lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
    return sha.stdout.strip(), bool(lines), len(lines), None


def _import_torch():
    """torch or None. Kept separate so a test can hand in a fake."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _gpu(torch_mod) -> tuple[str | None, str | None, str | None]:
    """`(gpu_name, cuda_version, reason)` from an imported torch, or the reason."""
    if torch_mod is None:
        return None, None, "torch not importable"
    try:
        if not torch_mod.cuda.is_available():
            return None, None, "no CUDA"
        name = torch_mod.cuda.get_device_name(torch_mod.cuda.current_device())
    except Exception as exc:                            # noqa: BLE001
        # A driver that is present but unusable is not a card identity.
        return None, None, f"CUDA query failed: {exc.__class__.__name__}"
    cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    return name, (cuda or None), None


def _nvidia_smi_driver_version() -> tuple[str | None, str | None]:
    """`(driver_version, reason)` from nvidia-smi; None with the reason otherwise."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return None, "nvidia-smi not on PATH"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"nvidia-smi failed: {exc.__class__.__name__}"
    if out.returncode != 0:
        return None, f"nvidia-smi exited {out.returncode}"
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return None, "nvidia-smi printed nothing"
    return lines[0], None


def _package_version(dist: str) -> tuple[str | None, str | None]:
    """`(version, reason)` via importlib.metadata. Looked up at call time so a
    test can monkeypatch `importlib.metadata.version`."""
    try:
        return importlib.metadata.version(dist), None
    except importlib.metadata.PackageNotFoundError:
        return None, "package not installed"
    except Exception as exc:                            # noqa: BLE001
        return None, f"metadata lookup failed: {exc.__class__.__name__}"


def _hostname() -> tuple[str | None, str | None]:
    try:
        name = socket.gethostname()
    except OSError as exc:
        return None, f"gethostname failed: {exc.__class__.__name__}"
    return (name or None), (None if name else "gethostname returned empty")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance_block(*, repo_root: Path | str | None = None, **known) -> Provenance:
    """Probe the environment, take the caller's facts, and NEVER raise for a
    missing one.

    `repo_root` is where git is asked; default is this package's repository.
    `known` supplies the caller-owned fields in `CALLER_FIELDS` (instrument,
    ridge, ridge_source, bandwidth, bandwidth_source, warmup_ms, iters,
    target_ms). A caller-owned field left out is None with the reason
    "not supplied by caller". A key not in `CALLER_FIELDS` is a TypeError: that
    is a misspelling at the call site, not something the machine did.
    """
    unknown = sorted(set(known) - set(CALLER_FIELDS))
    if unknown:
        raise TypeError(f"provenance_block() got unexpected field(s) {unknown}; "
                        f"caller-owned fields are {list(CALLER_FIELDS)}")

    missing: dict[str, str] = {}
    values: dict[str, Any] = {}

    sha, dirty, dirty_files, reason = _git(Path(repo_root) if repo_root else None)
    values["git_sha"], values["git_dirty"], values["git_dirty_files"] = sha, dirty, dirty_files
    if reason is not None:
        for name, val in (("git_sha", sha), ("git_dirty", dirty),
                          ("git_dirty_files", dirty_files)):
            if val is None:
                missing[name] = reason

    values["utc"] = _utc_now()

    host, reason = _hostname()
    values["hostname"] = host
    if reason:
        missing["hostname"] = reason

    torch_mod = _import_torch()
    gpu_name, cuda_version, reason = _gpu(torch_mod)
    values["gpu_name"], values["cuda_version"] = gpu_name, cuda_version
    if reason:
        missing["gpu_name"] = reason
        missing["cuda_version"] = reason
    elif cuda_version is None:
        missing["cuda_version"] = "torch.version.cuda is empty"

    if gpu_name is None:
        values["driver_version"] = None
        missing["driver_version"] = missing["gpu_name"]
    else:
        drv, reason = _nvidia_smi_driver_version()
        values["driver_version"] = drv
        if reason:
            missing["driver_version"] = reason

    values["python"] = platform.python_version()

    for field_name, dist in PACKAGE_FIELDS.items():
        version, reason = _package_version(dist)
        values[field_name] = version
        if reason:
            missing[field_name] = reason

    for name in CALLER_FIELDS:
        if name in known and known[name] is not None:
            values[name] = known[name]
        else:
            values[name] = None
            missing[name] = ("supplied as None" if name in known
                             else "not supplied by caller")
    for number, source in (("ridge", "ridge_source"), ("bandwidth", "bandwidth_source")):
        if values[number] is not None and values[source] is None:
            missing[source] = "supplied without a source"

    return Provenance(missing=missing, **values)


# --------------------------------------------------------------------------
# run ids
# --------------------------------------------------------------------------

#: Visible part of the id is capped so a directory name stays readable; the
#: hash carries the full key regardless.
_VISIBLE_MAX = 96


def card_slug(card: str) -> str:
    """`"NVIDIA H200"` -> `"nvidia_h200"`, the same rule the calibration file
    stems use, so a run directory and its calibration sort together."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(card).lower()).strip("_")
    if not slug:
        raise NoCard(f"card {card!r} has no alphanumeric characters to slug")
    return slug


def _canonical(value: Any) -> Any:
    """A JSON-stable form of a knob value. Tuples become lists, sets sorted
    lists, Paths strings; anything else must already be JSON-native."""
    if value is None:
        raise UnresolvedKnob("a swept knob is None; resolve it or leave it out")
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"swept knob value {value!r} of type {type(value).__name__} "
                    "is not JSON-native; convert it before naming the run")


def _visible(value: Any) -> str:
    canon = _canonical(value)
    if isinstance(canon, list):
        text = "_".join(str(v) for v in canon)
    elif isinstance(canon, dict):
        text = "_".join(f"{k}{v}" for k, v in canon.items())
    else:
        text = str(canon)
    return re.sub(r"[^a-z0-9.]+", "_", text.lower()).strip("_")


def run_id(*, card: str, **swept) -> str:
    """`<card_slug>-<k1><v1>-<k2><v2>-...-<hash8>`, deterministic, card first.

    Raises `NoCard` on a None or empty card and `UnresolvedKnob` on a None
    value. Knobs are sorted by name before rendering and hashing, so the id is
    identical whatever order the caller passed them in, and any change to any
    value changes the hash. The visible part is truncated at `_VISIBLE_MAX`
    characters; the hash is over the full key.
    """
    if card is None or not str(card).strip():
        raise NoCard("run_id() needs the card; an id without it collides across pods "
                     "(see the module docstring, collision 2)")
    slug = card_slug(card)
    items = sorted((str(k), _canonical(v)) for k, v in swept.items())
    key = json.dumps({"card": slug, "swept": items}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    parts = [slug]
    for k, v in items:
        parts.append(f"{re.sub(r'[^a-z0-9]+', '_', k.lower()).strip('_')}{_visible(v)}")
    visible = "-".join(parts)
    if len(visible) > _VISIBLE_MAX:
        visible = visible[:_VISIBLE_MAX].rstrip("-_")
    return f"{visible}-{digest}"
