r"""vLLM 0.27.1's fused-MoE tile lookup, run offline, for rows that never recorded one.

WHY THIS EXISTS. All ten published arms are schema v3, so not one of them
carries a `tile_block_m`. Every tile statement in `docs/FINDINGS.md` about those
arms -- the mixtral 64-to-128 step at T=256 that explains its cross-card
deviation, the `BLOCK_M=16` cap that decides whether a decode-tuned kernel can
ever reach its compute roof, the tile steps the crossing detector reads instead
of the ridge -- is currently a sentence in prose that was worked out by hand once
and cannot be rechecked. This module makes that derivation a function, so it can
be tested, reused and disagreed with.

WHAT IT IS NOT. Nothing here observes anything. `resolve_tile` answers "what
would vLLM v0.27.1 have loaded, given the `(E, N, dtype, gpu_name)` this row
records and the token count it was measured at", and that is an argument from
upstream's source plus a snapshot of upstream's config tree, not a measurement of
the kernel that ran. The distinction is not pedantry: a `BLOCK_SIZE_M = 128` that
nothing had measured sat unchallenged in an analysis for days and produced a
confident wrong mechanism for mixtral. So

  - every numeric field on `DerivedTile` ends in `_derived`,
  - none of those names is a schema column, asserted at import time by
    `_NAMES_CANNOT_COLLIDE_WITH_SCHEMA`, so a derived value cannot be written
    into a v4 observed column by a typo or a `**asdict()`,
  - `provenance` reads `vllm_tuned_derived` / `vllm_default_derived` rather than
    the schema's `vllm_tuned` / `vllm_default`, so a derived row and an observed
    row never group together in a report that keys on the source string,
  - and `DerivedTile.observed` is a field, permanently False, so a consumer that
    wants to refuse derived values has something to test rather than having to
    know which module the object came from.

`moe.bench.schema.tile_field` is the reader for the OBSERVED columns and raises
on a v3 row. This module is the other half of that answer: what may be said about
a v3 row anyway, said as a derivation.

THE THREE PIECES OF v0.27.1 BEHAVIOUR THIS REPRODUCES, each one able to
mis-predict a filename or a tile if guessed rather than read:

1. `get_config_file_name` builds `E={E},N={N},device_name={dev}` with an optional
   `,dtype=` and `,block_shape=` selector. `N` is w2's LAST dim, the per-shard
   intermediate size, because `try_get_optimal_moe_config` unpacks
   `E, _, N = w2_shape`. bf16 carries NO dtype suffix, because
   `_get_config_dtype_str` returns None for it.
2. The device selector is `re.sub(r"[\s/]+", "_", name)` -- runs of whitespace
   AND a slash, which a plain `.replace(" ", "_")` gets wrong -- followed by a
   fold of the whole H200 family onto `NVIDIA_H200` when any underscore-separated
   token is exactly `H200`.
3. THE LOOKUP TAKES THE NEAREST KEY, NOT THE FLOOR:
   `configs[min(configs.keys(), key=lambda x: abs(x - M))]`, with `M` the TOKEN
   count entering the layer (`M = num_tokens` in `fused_experts_impl`), never
   rows and never rows-per-expert. A floor reading of that lookup attributes a
   tile the row never ran: mixtral at M=113 resolves the key 128 and not 96, and
   at M=787 it resolves 1024 and not 512.

Ties go to the key that appears FIRST in the file, because `min` keeps the first
minimum and `get_moe_configs` builds its dict by `{int(key): val for ...}` over
`json.load`, which preserves file order. That is not a detail either: mixtral's
M=112 and M=192 are both exact ties, and both resolve DOWN.

The snapshot and how to extend it are described in
`moe/bench/hardware/vllm_configs/SNAPSHOT.md`.
"""
from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..quant import FP8_DTYPES
from ..spec import MODEL_CONFIGS
from . import schema as SC

#: The tag every file in the snapshot came from, and the only version this
#: module claims to reproduce. Carried on every `DerivedTile` so a derived value
#: quoted somewhere else still says which vLLM it describes.
VLLM_TAG = "v0.27.1"

SNAPSHOT_DIR = Path(__file__).resolve().parent / "hardware" / "vllm_configs"

#: All 327 config file names shipped at `VLLM_TAG`. Present so that "no tuned
#: file exists for this shape" is DECIDABLE rather than being confused with "the
#: file exists upstream and nobody vendored it". See SNAPSHOT.md.
SHIPPED_LISTING = SNAPSHOT_DIR / "SHIPPED_FILE_NAMES.txt"

#: `provenance` values. Deliberately NOT `schema.TILE_SOURCES` members. A report
#: that groups by source must not be able to pool a derived row with an observed
#: one, and equal strings are how that pooling would happen silently.
DERIVED_TUNED = "vllm_tuned_derived"
DERIVED_DEFAULT = "vllm_default_derived"

#: The implementations whose tile vLLM's lookup describes. SGLang runs its own
#: config tree and torch's grouped GEMM is CUTLASS, whose tile is fixed at 64 by
#: the wgmma instruction shape (claim C1) and owes nothing to this file.
VLLM_IMPLS = frozenset({"vllm_fused_experts", "__pipeline__:vllm_fused_experts"})


class TileNotDerivable(LookupError):
    """This row's tile cannot be derived, and no default may stand in for it.

    Raised rather than returning a plausible config for the same reason
    `schema.TileConfigUnrecorded` is: the failure mode being defended against is
    a number that reads as measured, and a resolver that answers for a kernel it
    does not model produces exactly that.
    """


class SnapshotMissing(FileNotFoundError):
    """A tuned file DOES ship for this shape and is not in the snapshot.

    The one case that must never silently become "took the fallback ladder".
    Both states look like "not on disk", they give different tiles, and only one
    of them is true of the run. `SHIPPED_FILE_NAMES.txt` is what tells them
    apart.
    """


@dataclass(frozen=True)
class DerivedTile:
    """What vLLM v0.27.1 WOULD have resolved. Derived throughout.

    Every int ends in `_derived` and none of the names is a schema column, so
    this object cannot be splatted into a `Row`. `config_key_derived` is 0 when
    the ladder was taken, which is honest rather than unrecorded: there was no
    tuned file to key into, and `describe()` says so in words.
    """

    block_m_derived: int
    block_n_derived: int
    block_k_derived: int
    group_m_derived: int
    num_warps_derived: int
    num_stages_derived: int
    #: DERIVED_TUNED or DERIVED_DEFAULT.
    provenance: str
    #: The exact filename vLLM would have tried to open, whether or not it exists.
    config_file: str
    #: The nearest key selected inside that file, 0 when there is no file.
    config_key_derived: int
    vllm_tag: str
    #: Permanently False. A field rather than an absence so that a consumer can
    #: write `if not tile.observed: ...` instead of having to know that anything
    #: from this module is derived.
    observed: bool = False

    def __post_init__(self) -> None:
        """Make "permanently False" true, rather than true only by default.

        The field above is documented as permanently False and, until this
        guard, was merely DEFAULTED to False: `DerivedTile(**{**asdict(tile),
        "observed": True})` built a derived tile that claimed to have been
        measured, and every other defence in this module (the `_derived`
        suffixes, the import-time collision check, the provenance strings kept
        out of `schema.TILE_SOURCES`) is downstream of a consumer trusting
        `tile.observed`. A guarantee a caller can switch off with a keyword is
        not a guarantee.
        """
        if self.observed:
            raise ValueError(
                "DerivedTile.observed is permanently False. Everything this "
                "module produces is derived from vLLM's source plus the "
                "recorded gpu_name, never measured. A tile a run actually "
                "observed lives in the schema v4 tile_* columns and is read "
                "with schema.tile_field.")

    @property
    def from_tuned_file(self) -> bool:
        return self.provenance == DERIVED_TUNED

    def describe(self) -> str:
        """One line for a report, which always says the word DERIVED."""
        where = (f"tuned {self.config_file} key {self.config_key_derived}"
                 if self.from_tuned_file
                 else f"fallback ladder, no {self.config_file}")
        return (f"BLOCK_M={self.block_m_derived} BLOCK_N={self.block_n_derived} "
                f"BLOCK_K={self.block_k_derived} GROUP_M={self.group_m_derived} "
                f"warps={self.num_warps_derived} stages={self.num_stages_derived} "
                f"[DERIVED from vLLM {self.vllm_tag}: {where}]")

    def as_columns(self) -> dict[str, int | str]:
        """A flat dict for a dataframe or a CSV of DERIVED values.

        Safe to merge beside real columns precisely because the key names are
        checked at import against `schema.COLUMNS` and cannot collide.
        """
        return asdict(self)


# Structural, not a test, because a test can be skipped and this cannot: if a
# later edit renames a field to a schema column name, importing this module
# fails rather than a derived value quietly becoming writable into an observed
# column.
_NAMES_CANNOT_COLLIDE_WITH_SCHEMA = [f.name for f in fields(DerivedTile)]
_collisions = sorted(set(_NAMES_CANNOT_COLLIDE_WITH_SCHEMA) & set(SC.COLUMNS))
if _collisions:                                              # pragma: no cover
    raise AssertionError(
        f"DerivedTile fields {_collisions} are also schema columns. Every value "
        f"this module produces is DERIVED and must not be writable into an "
        f"observed column; rename the field.")


# --------------------------------------------------------------------------
# the filename
# --------------------------------------------------------------------------

def device_selector(gpu_name: str) -> str:
    r"""`get_device_name_as_file_name` plus the H200 family fold.

    `re.sub(r"[\s/]+", "_", ...)` collapses RUNS of whitespace and rewrites a
    slash; `.replace(" ", "_")` differs on both. The fold matches a whole
    underscore-separated token, so `NVIDIA H200 NVL` reads the plain H200
    configs while `NVIDIA GH200 480GB` does not.

    Takes the `gpu_name` a CSV row records, which is torch's name verbatim.
    """
    name = re.sub(r"[\s/]+", "_", gpu_name.strip())
    if "H200" in name.split("_"):
        name = "NVIDIA_H200"
    return name


def config_dtype_selector(dtype: str) -> str | None:
    """The `dtype=` selector `_get_config_dtype_str` returns for this cell.

    None for bf16, which is why the bf16 files carry no suffix at all. The two
    fp8 formats this harness can sweep both reach `use_fp8_w8a8=True` in the
    span, so both select `fp8_w8a8`.

    Raises on anything else rather than defaulting to None. A silent None would
    name the bf16 file for a quantised cell, which is a real file that a real
    lookup would find, so the mistake would resolve to a plausible tile instead
    of failing.
    """
    if dtype in FP8_DTYPES:
        return "fp8_w8a8"
    if dtype in ("bf16", "fp16", "fp32"):
        return None
    raise TileNotDerivable(
        f"dtype {dtype!r} has no known vLLM config selector; the int4/int8 "
        f"paths also rewrite N and take a different default branch, so nothing "
        f"here may guess for them")


def config_file_name(num_experts: int, intermediate_n: int,
                     dtype_selector: str | None = None,
                     device_name: str = "NVIDIA_H200",
                     block_shape: list[int] | None = None) -> str:
    """`get_config_file_name`, reimplemented from the v0.27.1 source.

    `intermediate_n` is w2's LAST dim, i.e. the PER-SHARD intermediate size,
    because that is what `E, _, N = w2_shape` unpacks. Passing the full
    unsharded width here names a file for a deployment that was never run.
    """
    dtype_part = "" if not dtype_selector else f",dtype={dtype_selector}"
    block_part = ("" if not block_shape or not all(block_shape)
                  else f",block_shape={block_shape}").replace(" ", "")
    return (f"E={num_experts},N={intermediate_n},device_name={device_name}"
            f"{dtype_part}{block_part}.json")


@functools.lru_cache(maxsize=1)
def shipped_file_names() -> frozenset[str]:
    """Every config file name shipped at `VLLM_TAG`, from the snapshot listing."""
    text = SHIPPED_LISTING.read_text()
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def ships(name: str) -> bool:
    """Does vLLM v0.27.1 ship a tuned file under this name?"""
    return name in shipped_file_names()


# --------------------------------------------------------------------------
# the tuned lookup
# --------------------------------------------------------------------------

@functools.cache
def tuned_configs(name: str) -> tuple[tuple[int, tuple[tuple[str, int], ...]], ...] | None:
    """The tuned ladder for `name`, IN FILE ORDER, or None when none ships.

    Returned as nested tuples rather than a dict because `lru_cache` hands the
    same object to every caller and a dict would let one of them mutate the
    snapshot for all the others -- which is the failure vLLM itself guards
    against by deep-copying the config it returns, since `fused_experts_impl`
    overwrites `BLOCK_SIZE_M` downstream for the tail path.

    FILE ORDER IS LOAD-BEARING. `min` keeps the first of several equal minima and
    the keys are inserted in the order `json.load` yields them, so an exact tie
    resolves to whichever key the file lists first. mixtral has two such ties in
    the measured range, M=112 and M=192, and both go DOWN.

    Raises `SnapshotMissing`, never returns None, when the name ships upstream
    but is not vendored. Those two states are indistinguishable on disk and give
    different tiles.
    """
    path = SNAPSHOT_DIR / name
    if not path.exists():
        if ships(name):
            raise SnapshotMissing(
                f"vLLM {VLLM_TAG} ships {name} but it is not in "
                f"{SNAPSHOT_DIR}. Resolving it as the fallback ladder would "
                f"report a tile the measured run did not use. Add it with the "
                f"two-step recipe in {SNAPSHOT_DIR / 'SNAPSHOT.md'}.")
        return None
    raw = json.loads(path.read_text())
    raw.pop("triton_version", None)          # get_moe_configs drops this key
    return tuple((int(key), tuple(sorted(val.items()))) for key, val in raw.items())


def nearest_key(keys: tuple[int, ...], m: int) -> int:
    """`min(configs.keys(), key=lambda x: abs(x - M))`, ties to the first key.

    Written out rather than open-coded at the call site because the whole point
    of it is that it is NOT a floor, and a reader has to be able to see that in
    one place. `min` is stable in the sense that matters here: on equal scores it
    keeps the earliest item of the iterable, which is file order.
    """
    if not keys:
        raise ValueError("no keys to choose from")
    return min(keys, key=lambda x: abs(x - m))


# --------------------------------------------------------------------------
# the fallback ladder
# --------------------------------------------------------------------------

def default_config(m: int, num_experts: int,
                   dtype_selector: str | None = None) -> dict[str, int]:
    """`get_default_config`'s general branch, verbatim from the v0.27.1 source.

    This is what 12 of this study's 16 `(model x card x dtype)` combinations
    actually ran, so it is not a fallback in any practical sense -- it is the
    common case, and the four A100 cells that claim C5 turns on are all here.

    The M ladder is the part every write-up quotes:

        M <= 32 -> 16,   M <= 96 -> 32,   M <= 512 -> 64,   else 128

    The rest of the branch is quoted much less and matters as much. `GROUP_SIZE_M`
    is 16 only when `M // E > 128` and 1 otherwise, so on a many-expert model it
    is pinned at 1 across the entire measured range: deepseek-v3 at E=256 needs
    M > 32768 to leave it, and this study's largest cell is 8192. That is
    directly relevant to whether `alpha` can be a scalar, since GROUP_SIZE_M is
    the swizzle width and therefore how many M-tiles share a weight block in L2.

    NOT REIMPLEMENTED, and raising rather than guessing:
      - the `fp8_w8a8` + `block_shape` branch, which picks a different tile
        entirely. No cell here uses a block shape (`vllm_quant_spec` passes
        `block_shape=None`), and real fp8 MoE serving usually does, so this is a
        scope limit of the STUDY that the resolver should not paper over.
      - the int4/int8 wna16 branches, which also double N in the lookup.
      - `VLLM_BATCH_INVARIANT`, which returns a fixed 64/64/32/8 before any of
        this. No sweep here sets it.
    """
    if m <= 0:
        raise ValueError("M must be positive")
    if m <= 32:
        block_m = 16
    elif m <= 96:
        block_m = 32
    elif m <= 512:
        block_m = 64
    else:
        block_m = 128
    block_n = 64 if m <= 64 else 128
    block_k = 128 if dtype_selector == "fp8_w8a8" or m <= 64 else 64
    tokens_per_expert = m // max(num_experts, 1)
    group_m = 16 if tokens_per_expert > 128 else 1
    num_warps = 4 if m <= 128 else 8
    num_stages = 4 if m <= 32 else 3
    return {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group_m,
            "num_warps": num_warps, "num_stages": num_stages}


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def resolve_tile(num_experts: int, intermediate_n: int, dtype: str,
                 gpu_name: str, num_tokens: int,
                 block_shape: list[int] | None = None) -> DerivedTile:
    """The tile vLLM v0.27.1 would have resolved for this cell. DERIVED.

    `num_tokens` is `M`, the tokens entering the layer. Not rows (`T*k`), not
    rows per expert. `fused_experts_impl` sets `M = hidden_states.size(0)` and
    hands that straight to `try_get_optimal_moe_config`, so a resolver fed
    `rows` would step up the ladder `k` times too early.
    """
    if num_tokens <= 0:
        raise TileNotDerivable(
            f"num_tokens={num_tokens} is not a batch vLLM could have been "
            f"called with, so there is no lookup to reproduce")
    if block_shape:
        raise TileNotDerivable(
            "block_shape sends the lookup down a branch this module does not "
            "reimplement; no cell in this study passes one")
    selector = config_dtype_selector(dtype)
    name = config_file_name(num_experts, intermediate_n, selector,
                            device_selector(gpu_name))
    entries = tuned_configs(name)
    if entries is None:
        cfg = default_config(num_tokens, num_experts, selector)
        provenance, key = DERIVED_DEFAULT, 0
    else:
        chosen = nearest_key(tuple(k for k, _ in entries), num_tokens)
        cfg = dict(next(dict(v) for k, v in entries if k == chosen))
        provenance, key = DERIVED_TUNED, chosen
    return DerivedTile(
        block_m_derived=int(cfg["BLOCK_SIZE_M"]),
        block_n_derived=int(cfg["BLOCK_SIZE_N"]),
        block_k_derived=int(cfg["BLOCK_SIZE_K"]),
        group_m_derived=int(cfg["GROUP_SIZE_M"]),
        num_warps_derived=int(cfg["num_warps"]),
        num_stages_derived=int(cfg["num_stages"]),
        provenance=provenance,
        config_file=name,
        config_key_derived=key,
        vllm_tag=VLLM_TAG,
    )


def resolve_tile_for_row(row: Mapping) -> DerivedTile:
    """`resolve_tile` from a published CSV row's own columns.

    Every input is read off the row or off `MODEL_CONFIGS`, never restated here,
    so a derivation cannot agree with this file while disagreeing with the
    harness that produced the row. `(E, N)` comes off `w2_shape` for the same
    reason vLLM's own lookup does.

    Refuses a row whose `impl` is not a vLLM Triton span. A SGLang row has a tile
    and this file does not know it; a torch row's tile is CUTLASS's 64, observed
    under C1 and not derived from anything here. Answering for either would be
    the exact substitution this module exists to prevent.
    """
    impl = str(row.get("impl", ""))
    if impl not in VLLM_IMPLS:
        raise TileNotDerivable(
            f"impl {impl!r} does not resolve a vLLM fused-MoE config: "
            f"sglang_* uses its own tuned tree, torch_* is CUTLASS with the "
            f"tile fixed at 64 by the wgmma shape (C1). Derivable impls are "
            f"{sorted(VLLM_IMPLS)}.")
    model = str(row.get("model", ""))
    if model not in MODEL_CONFIGS:
        raise TileNotDerivable(f"model {model!r} is not in MODEL_CONFIGS")
    gpu = str(row.get("gpu_name", ""))
    if not gpu:
        raise TileNotDerivable("the row records no gpu_name, so there is no "
                               "device selector and no filename to look up")
    num_experts, _, intermediate_n = MODEL_CONFIGS[model].w2_shape
    return resolve_tile(num_experts, intermediate_n, str(row.get("dtype", "")),
                        gpu, int(float(row.get("num_tokens") or 0)))


def disagreement_with_observed(row: Mapping) -> str | None:
    """For a v4 row: does the derivation match what the run actually recorded?

    None when they agree, when the row is v3 (nothing observed to compare
    against), or when the row is not a vLLM span. A string naming both values
    when they differ.

    This is the check that converts the whole module from an argument into a
    testable claim, and it can only be run on a pod. Until a v4 arm exists every
    call returns None, which is the honest answer and not a pass.
    """
    if not SC.has_tile_config(row):
        return None
    try:
        derived = resolve_tile_for_row(row)
    except TileNotDerivable:
        return None
    observed = int(SC.tile_field(row, "tile_block_m"))
    if observed == derived.block_m_derived:
        return None
    return (f"{row.get('model')} T={row.get('num_tokens')} "
            f"{row.get('dtype')} on {row.get('gpu_name')}: the row OBSERVED "
            f"BLOCK_M={observed}, this module DERIVES "
            f"{derived.block_m_derived} ({derived.describe()}). The observed "
            f"value wins; the derivation is wrong or the run was not vLLM "
            f"{VLLM_TAG}.")


# --------------------------------------------------------------------------
# the census: every published row's derived tile, as a command
# --------------------------------------------------------------------------

def census(rows: Iterable[Mapping]) -> dict[tuple, dict[int, int]]:
    """`{(model, gpu, dtype, provenance): {BLOCK_M: rows}}` over vLLM spans.

    The deliverable of this module in one table: before it, not one of the
    36,000-odd vLLM rows in `results/published/` could say which tile it ran.
    Rows whose tile is not derivable are skipped rather than bucketed under a
    default, and the skip shows up as a missing row rather than as a zero.
    """
    out: dict[tuple, dict[int, int]] = {}
    for row in rows:
        try:
            tile = resolve_tile_for_row(row)
        except TileNotDerivable:
            continue
        key = (str(row.get("model", "")), str(row.get("gpu_name", "")),
               str(row.get("dtype", "")), tile.provenance)
        bucket = out.setdefault(key, {})
        bucket[tile.block_m_derived] = bucket.get(tile.block_m_derived, 0) + 1
    return out


def _main(argv: list[str]) -> int:
    """`python -m moe.bench.tile_resolve results/published/*/run_*.csv`.

    Prints the derived-tile census and, for any v4 row, whether the derivation
    disagrees with what the run observed. An entry point rather than a script
    under `scripts/` for the same reason `published.py` has one: the rule and the
    table it produces should not be able to drift apart.
    """
    from .schema import read_csv
    paths = [Path(a) for a in argv]
    if not paths:
        print(f"usage: python -m moe.bench.tile_resolve <csv> ...\n"
              f"prints the tile vLLM {VLLM_TAG} WOULD have resolved for every "
              f"vLLM row. Every value is DERIVED.")
        return 2
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_csv(path))
    print(f"# DERIVED tile per published row, vLLM {VLLM_TAG}")
    print()
    print("Nothing here is observed. Each BLOCK_M is what vLLM's lookup would")
    print("return for the (E, N, dtype, gpu_name) the row records, at the row's")
    print("own token count.")
    print()
    print("| model | gpu | dtype | source | BLOCK_M -> rows |")
    print("|---|---|---|---|---|")
    for key in sorted(census(rows)):
        counts = census(rows)[key]
        spread = ", ".join(f"{bm}: {n}" for bm, n in sorted(counts.items()))
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {spread} |")
    disagreements = [d for d in (disagreement_with_observed(r) for r in rows) if d]
    print()
    if disagreements:
        print(f"## {len(disagreements)} rows OBSERVED a tile this module derives "
              f"differently")
        for line in disagreements[:20]:
            print(f"- {line}")
        return 1
    print("No row observed a tile to check the derivation against. Every input is")
    print("schema v3, which is exactly why this module had to be written, and it")
    print("is NOT a pass: nothing here has been validated against a run.")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
