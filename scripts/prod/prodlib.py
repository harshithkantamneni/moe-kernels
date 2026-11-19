"""Shared pieces of the production-check kit (scripts/prod/*). Pure Python + numpy: importable on the laptop.

What lives here, because more than one kit script needs the same answer:
  - the two models the kit serves (ERNIE-4.5-21B-A3B, the blind model; OLMoE-1B-7B, the smoke model), as MoEConfig
    objects built HERE (moe/spec.py is sha-pinned in >= 10 registrations and is not edited; v8 adds ERNIE to its own
    registry at runtime, KIT-SPEC / CRITIQUE B3);
  - vLLM 0.27.1's default tile at a padded T (moe.bench.tile_resolve.default_config, verbatim upstream), the tile
    label, the branch (decode T<=128 at the 1710 MHz lock, prefill T>128 at 1410, CRITIQUE B1) and the strata the
    scorer reports separately (CRITIQUE M4, M5);
  - the CUDA-graph capture sizes and the padded T a step runs at (CAPTURE.md);
  - the capture ring's on-disk format and its decoder (moe_capture.py writes, nsys_extract/score/replay read);
  - THE ONE binding to the v8 pricer (`price_v8`, v8's registered interface: {"model": MoEConfig, "T", "counts",
    "ids"} plus, on a prefill tile, the three constants v8 requires: the board's 128x128 sag par and the registered
    64x128 par and K1 knee, `PREFILL_REGISTERED`);
  - THE registered roofline rival (`roofline_ms`): the v7 decode blind's form and frozen constants, one place;
  - the pinned Hugging Face revisions of the served models (`HF_REVISIONS`).

    python scripts/prod/prodlib.py price-check --model ernie-4.5-21b-a3b     # one real decode + one prefill call
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
for _p in (str(REPO), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe.bench.tile_resolve import default_config  # noqa: E402
from moe.spec import MODEL_CONFIGS, MoEConfig  # noqa: E402

# --------------------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------------------

#: ERNIE-4.5-21B-A3B-PT (PLAN.md "Model under test"): E 64 routed (+2 shared, a separate dense MLP, not in the
#: measured call), top-6, H 2560, expert I 1536, 28 layers of which 1..27 are MoE. ONE definition: v8's own
#: (r3_timing_model_v8.ERNIE, registered into MODEL_CONFIGS in-process by v8.register_model), reused here so the
#: config the kit sends v8 is the one v8 registers (review 1, BLOCKER 1). serve_capture still diffs these numbers
#: against the downloaded config.json before the first step (G-CFG) and stops on a mismatch.
from r3_timing_model_v8 import ERNIE  # noqa: E402


@dataclass(frozen=True)
class ServedModel:
    key: str
    cfg: MoEConfig
    hf_repo: str
    moe_layers: int
    #: config.json keys that must equal cfg's numbers (G-CFG), as (json key, cfg attribute)
    hf_keys: tuple


MODELS = {
    "ernie-4.5-21b-a3b": ServedModel("ernie-4.5-21b-a3b", ERNIE, "baidu/ERNIE-4.5-21B-A3B-PT", 27,
                                     (("hidden_size", "hidden_size"), ("moe_intermediate_size", "intermediate_size"),
                                      ("moe_num_experts", "num_experts"), ("moe_k", "top_k"),
                                      ("num_hidden_layers", "num_layers"))),
    "olmoe-1b-7b": ServedModel("olmoe-1b-7b", MODEL_CONFIGS["olmoe-1b-7b"], "allenai/OLMoE-1B-7B-0924", 16,
                               (("hidden_size", "hidden_size"), ("intermediate_size", "intermediate_size"),
                                ("num_experts", "num_experts"), ("num_experts_per_tok", "top_k"),
                                ("num_hidden_layers", "num_layers"))),
}


def check_hf_config(model: ServedModel, hf: dict) -> list[str]:
    """G-CFG: every listed config.json number equals the kit's MoEConfig. ERNIE's moe_num_experts may be a list
    (one entry per modality); its first entry is the text experts."""
    bad = []
    for key, attr in model.hf_keys:
        got = hf.get(key)
        if isinstance(got, list):
            got = got[0] if got else None
        want = getattr(model.cfg, attr)
        if got != want:
            bad.append(f"config.json {key}={got!r}, the kit's {attr}={want!r}")
    return bad


# --------------------------------------------------------------------------------------------------------------
# tiles, branches, strata
# --------------------------------------------------------------------------------------------------------------

DECODE_MAX_T = 128
LOCKS = {"decode": 1710, "prefill": 1410}
#: M4's Hopper G-1 rule: w2 at G 1 with n_eff = T k / (E BM) >= HOPPER_G1_N is ill-posed (v8 refuses it)
HOPPER_G1_N = 6


def refused_window(cfg: MoEConfig) -> tuple[int, int] | None:
    """The T range v8 refuses by HOPPER_G1_N on vLLM 0.27.1's default ladder, PER MODEL (review 1, MINOR): the
    128x128 tile runs at G 1 while T // E <= 128, i.e. T <= 129 E - 1, and is refused once n_eff >= 6, i.e.
    T >= ceil(6 E 128 / k). ERNIE (E 64, k 6): 8192..8255; OLMoE (E 64, k 8): 6144..8255. None when empty."""
    lo = -(-HOPPER_G1_N * cfg.num_experts * 128 // cfg.top_k)
    lo = max(lo, 513)                                   # the 128x128 tile starts at T 513 (E 64 ladder)
    hi = 129 * cfg.num_experts - 1
    return (lo, hi) if lo <= hi else None


#: ERNIE's window, the registered one (CRITIQUE M5); every other model reads refused_window(cfg)
REFUSED_WINDOW = refused_window(ERNIE)
#: moe_align_block_size's small-batch kernel: numel < 1024 and E <= 64 (UPSTREAM_DECODE rule SMALL_BATCH)
SMALL_BATCH_NUMEL = 1024


def tile_conf(T: int, E: int) -> dict:
    return default_config(int(T), int(E))


def tile_label(T: int, E: int) -> str:
    """'BMxBNxBK w<warps> s<stages>' plus ' g16' when GROUP_SIZE_M is 16 (T // E > 128, e.g. T >= 8256 at E 64)."""
    c = tile_conf(T, E)
    lab = f"{c['BLOCK_SIZE_M']}x{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']}w{c['num_warps']}s{c['num_stages']}"
    return lab + (" g16" if c["GROUP_SIZE_M"] == 16 else "")


#: the seven tiles a serving run at E=64 lands on, in T order (KIT-SPEC 2 / CRITIQUE M9)
TILE_BINS = (("16x64x128w4s4", 1, 32), ("32x64x128w4s3", 33, 64), ("32x128x64w4s3", 65, 96),
             ("64x128x64w4s3", 97, 128), ("64x128x64w8s3", 129, 512), ("128x128x64w8s3", 513, 8255),
             ("128x128x64w8s3 g16", 8256, 10 ** 9))


def tile_bin(T: int) -> str:
    for lab, lo, hi in TILE_BINS:
        if lo <= T <= hi:
            return lab
    raise ValueError(f"T {T} is not a token count")


def branch(T: int) -> str:
    return "decode" if int(T) <= DECODE_MAX_T else "prefill"


#: the stratum name of the refused window (ERNIE's bounds in the name: the registered window; another model's window
#: is its own refused_window(cfg), under the same name)
REFUSED_STRATUM = "refused_window_8192_8255"


def strata(T: int, top_k: int, E: int = 64) -> list[str]:
    """The strata a call belongs to beyond its tile (CRITIQUE M4, M5). The refused window is the model's own
    (refused_window, k-dependent)."""
    out = []
    T = int(T)
    if 129 <= T <= 512:
        out.append("band_64x128")
    if T > DECODE_MAX_T and T * top_k < SMALL_BATCH_NUMEL and E <= 64:
        out.append("small_batch_align_on_prefill_tile")
    lo = max(-(-HOPPER_G1_N * E * 128 // top_k), 513)
    if lo <= T <= 129 * E - 1:
        out.append(REFUSED_STRATUM)
    return out


# --------------------------------------------------------------------------------------------------------------
# CUDA-graph capture sizes and padding (CAPTURE.md; verified on the VM against the engine's own config)
# --------------------------------------------------------------------------------------------------------------

def capture_sizes(max_size: int = 512) -> list[int]:
    """vLLM 0.27.1's default capture sizes at max_num_seqs 1024: 1, 2, 4, 8..248 step 8, 256..max step 16."""
    s = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
    return [x for x in s if x <= max_size]


def padded_T(T: int, mode: str, sizes: list[int] | None = None) -> int:
    """The T a step's forward runs at: graphs pad up to the next capture size when T <= the largest; eager and
    T beyond the largest run at T exactly."""
    if mode == "eager":
        return int(T)
    sizes = sorted(sizes or capture_sizes())
    for s in sizes:
        if s >= T:
            return s
    return int(T)


# --------------------------------------------------------------------------------------------------------------
# counts, digests, geometry
# --------------------------------------------------------------------------------------------------------------

def counts_from_ids(ids, E: int) -> list[int]:
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    if a.size and (a.min() < 0 or a.max() >= E):
        raise ValueError(f"expert id out of range 0..{E - 1}: min {a.min()} max {a.max()}")
    return np.bincount(a, minlength=E).astype(int).tolist()


def digest(x) -> str:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def post_padded(counts, bm: int) -> int:
    """num_tokens_post_padded on the aligned path: sum ceil(c_e / BM) BM."""
    return int(sum(-(-int(c) // bm) * bm for c in counts if c))


def expected_grids(cfg: MoEConfig, T: int, counts) -> tuple[int, int]:
    """fused_moe_kernel's launch grid (w1, w2) for this call: r3_decode.geometry (R3's fused_moe_grid: cdiv(EM, BM)
    cdiv(N, BN) with vLLM's EM cut and MAXPAD cap at small numel, and the naive path's own row count). The nsys gate
    compares gridX against it (Triton launches a 1-D grid)."""
    return call_geometry(cfg, T, counts)["grid"]


def call_geometry(cfg: MoEConfig, T: int, counts) -> dict:
    """{"grid": (w1, w2), "path": vLLM's assignment path (naive, or the align kernel's small/large path)}."""
    import r3_decode as RD
    g = RD.geometry(cfg, int(T), list(counts))
    return {"grid": (int(g.grid[0]), int(g.grid[1])), "path": str(g.path)}


# --------------------------------------------------------------------------------------------------------------
# the capture ring's files (written by moe_capture.Capture.drain)
# --------------------------------------------------------------------------------------------------------------

RING_SCHEMA = "moe-kernels/prod-ring/1"
#: columns of a shard's `recs` array
REC_COLS = ("layer", "n", "offset", "seq")


class RingError(RuntimeError):
    """A ring shard that cannot be trusted (overflow, a gap, a bad offset): the run's capture FAILS."""


def decode_shards(shard_paths, top_k: int) -> list[dict]:
    """Every captured call in record order, across shards: {"g": global index, "shard", "layer", "n", "ids"
    (np.int16 [n, k])}. Raises RingError on an overflowed shard, a seq gap, or an offset that does not tile the
    ids array exactly (the k-th record's ids start where the (k-1)-th ended)."""
    out: list[dict] = []
    g = 0
    for p in sorted(shard_paths, key=lambda q: Path(q).name):
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            ids = z["ids"]
            recs = z["recs"]
        if meta.get("schema") != RING_SCHEMA:
            raise RingError(f"{p}: schema {meta.get('schema')!r}, not {RING_SCHEMA}")
        if meta.get("overflow"):
            raise RingError(f"{p}: the device ring overflowed ({meta.get('overflow')}): calls were lost")
        n_rec = int(meta["n_records"])
        if recs.shape != (n_rec, len(REC_COLS)):
            raise RingError(f"{p}: recs shape {recs.shape}, meta says {n_rec} records")
        if int(meta["n_ids"]) != ids.size:
            raise RingError(f"{p}: {ids.size} ids on disk, meta says {meta['n_ids']}")
        off = 0
        for i, (layer, n, o, seq) in enumerate(recs.tolist()):
            if seq != i:
                raise RingError(f"{p}: record {i} carries seq {seq} (a gap or a reorder)")
            if o != off:
                raise RingError(f"{p}: record {i} starts at {o}, the previous ended at {off}")
            cnt = int(n) * top_k
            if o + cnt > ids.size:
                raise RingError(f"{p}: record {i} runs past the ids array")
            out.append({"g": g, "shard": Path(p).name, "layer": int(layer), "n": int(n),
                        "ids": ids[o:o + cnt].reshape(int(n), top_k), "tag": meta.get("tag", "")})
            off += cnt
            g += 1
        if off != ids.size:
            raise RingError(f"{p}: records cover {off} ids of {ids.size}")
    return out


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------------------------
# the prefill branch's constants (review 1, BLOCKER 2: v8 has no defaults; every one is named here or measured)
# --------------------------------------------------------------------------------------------------------------

#: The two REGISTERED prefill constants, by value with their source (tests/test_prod_score.py holds each to its
#: source module): the 64x128 sag par is v7's prefill blind's registered one (v7pcommon.A2_PAR_64, A2's board: no
#: on-board 64x128 measurement is registered for this check), the knee round 4's SEEN K1 (round5_chain.K1_SEEN).
PREFILL_REGISTERED = {
    "sag64": (30.365, 300.0),
    "knee": (6.0, 2.75),
    "source": {"sag64": "scripts/scoring/v7pblind/v7pcommon.py A2_PAR_64 (v7 prefill blind, 2026-11-18)",
               "knee": "scripts/round5_chain.py K1_SEEN (round 4's SEEN K1)"},
}
#: The 128x128 par is MEASURED on the board: main.sh's sag1410 step (scripts/prod/sag1410.sh) runs the v7 prefill
#: blind's driver units (prelude at 1410 / 700 W, the OLMoE and Mixtral-8x7B f128 bytes pages, gate b3sag) and
#: v7pcommon.finalise_pars; its par.json is what `load_board_par` reads and the scorer passes as sag128.
#: The four registered reference pars (v7pcommon.REF_PARS: A2, C2, C4, B3's boards) give the DESCRIPTIVE
#: "no on-board calibration" variant: v8 at their elementwise median (`REF_PAR_MEDIAN`).
REF_PARS = {"A2": (42.02, 425.0), "C2": (41.59, 412.5), "C4": (54.25, 587.5), "B3": (59.815, 612.5)}
REF_PAR_MEDIAN = (float(np.median([p[0] for p in REF_PARS.values()])),
                  float(np.median([p[1] for p in REF_PARS.values()])))
PAR_SCHEMA = "moe-kernels/prod-sag-par/1"


def prefill_constants(sag128) -> dict:
    """The three constants a prefill call carries: the given 128x128 par and the registered 64x128 par and knee."""
    if sag128 is None:
        raise KitError("no 128x128 sag par: the prefill branch has no default (main.sh's sag1410 step measures it)")
    a, t = (float(x) for x in sag128)
    return {"sag128": (a, t), "sag64": tuple(PREFILL_REGISTERED["sag64"]), "knee": tuple(PREFILL_REGISTERED["knee"])}


def load_board_par(path) -> tuple[float, float]:
    """The board's 128x128 par from sag_par.py's par.json (the finaliser's record, its verdict PASS)."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != PAR_SCHEMA or doc.get("verdict") != "PASS":
        raise KitError(f"{path}: schema {doc.get('schema')!r}, verdict {doc.get('verdict')!r}: not a finalised par")
    a, t = doc["par"]
    return float(a), float(t)


# --------------------------------------------------------------------------------------------------------------
# the roofline rival (review 1, MAJOR 4): the v7 decode blind's registered form at its frozen constants
# --------------------------------------------------------------------------------------------------------------

ROOFLINE_REG = REPO / "docs" / "registered" / "2026-11-17-v7-decode-blind-gh200.json"
_ROOF: dict = {}


def roofline_constants() -> dict:
    """{"T0_GR_ms", "bw_gbps", "source"}: the decode blind's rival form T0_GR + A W_e / bw
    (scripts/scoring/decode/finalise_decode.py:141 -> r3_timing_model_v4d.roofline_ms), T0_GR the frozen CAL2
    constant of docs/registered/2026-11-17-v7-decode-blind-gh200.json (predictions.frozen.constants.T0_GR_ms) and bw
    v4d's card rate (r3_timing_model_v4d.CARD["bw_gbps"], the default roofline_ms reads). Applied to every call,
    both branches (the registered rival has one set of constants)."""
    if not _ROOF:
        import r3_timing_model_v4d as V4D
        fz = json.loads(ROOFLINE_REG.read_text())["predictions"]["frozen"]
        _ROOF.update(T0_GR_ms=float(fz["constants"]["T0_GR_ms"]), bw_gbps=float(V4D.CARD["bw_gbps"]),
                     source=f"{ROOFLINE_REG.relative_to(REPO)} predictions.frozen.constants.T0_GR_ms; "
                            "r3_timing_model_v4d.CARD['bw_gbps']; form r3_timing_model_v4d.roofline_ms")
    return dict(_ROOF)


def roofline_ms_A(cfg: MoEConfig, A: int, c: dict | None = None) -> float:
    """T0_GR + A W_e / bw, W_e = 3 H I x 2 bytes (r3_decode.expert_bytes, bf16), in v4d's operation order."""
    c = c or roofline_constants()
    return c["T0_GR_ms"] + int(A) * (3 * cfg.hidden_size * cfg.intermediate_size * 2) / (c["bw_gbps"] * 1e6)


def roofline_ms(cfg: MoEConfig, counts, c: dict | None = None) -> float:
    """The roofline of one call: A = the experts with at least one routed row."""
    return roofline_ms_A(cfg, sum(1 for v in counts if int(v)), c)


# --------------------------------------------------------------------------------------------------------------
# pinned Hugging Face revisions (review 1, MINOR): the model and tokenizer commit each run loads
# --------------------------------------------------------------------------------------------------------------

#: {model key: commit sha of the HF repo}. None = NOT PINNED: smoke.sh runs unpinned and prints the resolved sha
#: (serve_capture's manifest "hf_revision"), which is pasted here before the kit freezes; main.sh refuses an unpinned
#: model at its first step (revision-pinned). The same sha pins the tokenizer (prompts.py) and the engine's
#: model and tokenizer (serve_capture.py: LLM(revision=, tokenizer_revision=)).
HF_REVISIONS: dict = {"ernie-4.5-21b-a3b": None, "olmoe-1b-7b": None}


def hf_revision(model_key: str) -> str | None:
    return HF_REVISIONS.get(model_key)


# --------------------------------------------------------------------------------------------------------------
# THE v8 binding (one function; switch it here when v8 lands)
# --------------------------------------------------------------------------------------------------------------

PRICER_MODULE = "r3_timing_model_v8"


def price_v8(cfg: MoEConfig, T: int, counts, ids=None, *, sag128=None, module: str | None = None) -> dict:
    """{"ms": float | None, "tile": str, "regime": "decode" | "prefill", "why": str} from v8's price(). The
    interface (v8's own, scripts/r3_timing_model_v8.py price()): {"model": MoEConfig, "T" (padded), "counts", "ids"}
    and, on a prefill tile (T > 128), "sag128" (the board's par, required), "sag64" and "knee" (PREFILL_REGISTERED).
    A refusal is ms None with its why; anything else missing from the answer is a KitError, not a silent None."""
    V8 = importlib.import_module(module or PRICER_MODULE)
    call = {"model": cfg, "T": int(T), "counts": [int(c) for c in counts],
            "ids": None if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)}
    if int(T) > DECODE_MAX_T:
        call.update(prefill_constants(sag128))
    ans = V8.price(call)
    if not isinstance(ans, dict) or "ms" not in ans or "regime" not in ans or "tile" not in ans:
        raise KitError(f"{module or PRICER_MODULE}.price answered {ans!r}: not the registered interface")
    if ans["ms"] is None and not ans.get("why"):
        raise KitError(f"{module or PRICER_MODULE}.price refused T={T} without a why")
    return ans


def price_check(model_key: str, *, sag128=None, module: str | None = None, seed: int = 20261119) -> dict:
    """One real decode call (T 8) and one real prefill call (T 1024) of the served model through `price_v8`, on
    seeded uniform routing with its ids: both must price (review 1, BLOCKER 1: the contract the kit and v8 share).
    sag128 None: the registered reference median (REF_PAR_MEDIAN; the board's par is not measured yet at M0)."""
    cfg = MODELS[model_key].cfg
    out = {}
    for T in (8, 1024):
        rng = np.random.default_rng(seed + T)
        ids = np.argsort(rng.random((T, cfg.num_experts)), axis=1)[:, :cfg.top_k]
        ans = price_v8(cfg, T, counts_from_ids(ids, cfg.num_experts), ids, sag128=sag128 or REF_PAR_MEDIAN,
                       module=module)
        out[T] = {"ms": ans["ms"], "tile": ans["tile"], "regime": ans["regime"], "why": ans.get("why", "")}
        if ans["ms"] is None:
            raise KitError(f"price-check: {model_key} T {T} refused ({ans.get('why')})")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="prodlib: the kit's v8 contract check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("price-check")
    pc.add_argument("--model", required=True, choices=sorted(MODELS))
    pc.add_argument("--pricer", default=None)
    pc.add_argument("--sag", default=None, help="the board's par.json (default: the registered reference median)")
    a = ap.parse_args(argv)
    try:
        got = price_check(a.model, sag128=load_board_par(a.sag) if a.sag else None, module=a.pricer)
    except (KitError, ValueError, KeyError, ImportError) as exc:
        print(f"FAIL price-check: {type(exc).__name__}: {exc}")
        return 3
    print(json.dumps({str(k): v for k, v in got.items()}))
    print(f"PASS price-check {a.model}: decode T 8 {got[8]['ms']:.4f} ms ({got[8]['tile']}), prefill T 1024 "
          f"{got[1024]['ms']:.4f} ms ({got[1024]['tile']})")
    return 0


class KitError(RuntimeError):
    """A kit invariant broke: the step stops and says why (never a silent fallback)."""


if __name__ == "__main__":
    sys.exit(main())
