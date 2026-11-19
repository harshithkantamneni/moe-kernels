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
  - THE ONE binding to the v8 pricer (`price_v8`), so it can be switched in one place when v8 lands.
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
#: measured call), top-6, H 2560, expert I 1536, 28 layers of which 1..27 are MoE. verified=False: serve_capture
#: diffs these numbers against the downloaded config.json before the first step (G-CFG) and stops on a mismatch.
ERNIE = MoEConfig(name="ernie-4.5-21b-a3b", hidden_size=2560, intermediate_size=1536, num_experts=64, top_k=6,
                  gate_fn="softmax", norm_topk_prob=True, shared_experts=2, num_layers=28, first_moe_layer=1,
                  hf_repo="baidu/ERNIE-4.5-21B-A3B-PT", verified=False)


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
#: v8 refuses the HOPPER_G1_N window (CRITIQUE M5): reported with its share of MoE time
REFUSED_WINDOW = (8192, 8255)
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


def strata(T: int, top_k: int, E: int = 64) -> list[str]:
    """The strata a call belongs to beyond its tile (CRITIQUE M4, M5)."""
    out = []
    T = int(T)
    if 129 <= T <= 512:
        out.append("band_64x128")
    if T > DECODE_MAX_T and T * top_k < SMALL_BATCH_NUMEL and E <= 64:
        out.append("small_batch_align_on_prefill_tile")
    if REFUSED_WINDOW[0] <= T <= REFUSED_WINDOW[1]:
        out.append("refused_window_8192_8255")
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
    """fused_moe_kernel's launch grid (w1, w2) for this call: r3_decode.geometry (R3's fused_moe_grid, the EM cut
    in it). The nsys gate compares gridX against it (Triton launches a 1-D grid)."""
    import r3_decode as RD
    g = RD.geometry(cfg, int(T), list(counts))
    return int(g.grid[0]), int(g.grid[1])


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
# THE v8 binding (one function; switch it here when v8 lands)
# --------------------------------------------------------------------------------------------------------------

PRICER_MODULE = "r3_timing_model_v8"


def price_v8(cfg: MoEConfig, T: int, counts, ids=None, *, module: str | None = None) -> dict:
    """{"ms": float | None, "tile": str, "regime": "decode" | "prefill", "why": str} from v8's price(). The
    interface (owner, 2026-11-19): V8.price({"model_cfg", "T" (padded), "counts", "ids"}). A refusal is ms None
    with its why; anything else missing from the answer is a KitError, not a silent None."""
    V8 = importlib.import_module(module or PRICER_MODULE)
    ans = V8.price({"model_cfg": cfg, "T": int(T), "counts": [int(c) for c in counts],
                    "ids": None if ids is None else [int(x) for x in np.asarray(ids).reshape(-1)]})
    if not isinstance(ans, dict) or "ms" not in ans or "regime" not in ans or "tile" not in ans:
        raise KitError(f"{module or PRICER_MODULE}.price answered {ans!r}: not the registered interface")
    if ans["ms"] is None and not ans.get("why"):
        raise KitError(f"{module or PRICER_MODULE}.price refused T={T} without a why")
    return ans


class KitError(RuntimeError):
    """A kit invariant broke: the step stops and says why (never a silent fallback)."""
