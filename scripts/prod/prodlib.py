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
    #: config.json keys that must hold a literal value (G-CFG), as (json key, value): e.g. Qwen3's every-layer-MoE keys
    hf_literals: tuple = ()


#: Qwen3-30B-A3B (extras.sh X3, the SECOND BLIND MODEL, owner 2026-11-19): E 128 routed, top-8, H 2048, expert I 768,
#: 48 layers, every one MoE (decoder_sparse_step 1, mlp_only_layers []), no shared expert. Built HERE at runtime like
#: ERNIE (moe/spec.py is sha-pinned and is not edited; v8 registers it in-process from the MoEConfig price_v8 passes).
#: Checked 2026-11-19 against the Hub's config.json at the pinned revision (HF_REVISIONS): hidden_size 2048,
#: moe_intermediate_size 768, num_experts 128, num_experts_per_tok 8, num_hidden_layers 48, norm_topk_prob true,
#: decoder_sparse_step 1, mlp_only_layers [] (no shared_expert_intermediate_size key: vLLM 0.27.1's
#: Qwen3MoeSparseMoeBlock builds no shared expert, qwen3_moe.py:180-202, and routes through FusedMoEFactory).
QWEN3 = MoEConfig(name="qwen3-30b-a3b", hidden_size=2048, intermediate_size=768, num_experts=128, top_k=8,
                  gate_fn="softmax", norm_topk_prob=True, shared_experts=0, num_layers=48, first_moe_layer=0,
                  hf_repo="Qwen/Qwen3-30B-A3B", verified=True)


MODELS = {
    "ernie-4.5-21b-a3b": ServedModel("ernie-4.5-21b-a3b", ERNIE, "baidu/ERNIE-4.5-21B-A3B-PT", 27,
                                     (("hidden_size", "hidden_size"), ("moe_intermediate_size", "intermediate_size"),
                                      ("moe_num_experts", "num_experts"), ("moe_k", "top_k"),
                                      ("num_hidden_layers", "num_layers"))),
    "olmoe-1b-7b": ServedModel("olmoe-1b-7b", MODEL_CONFIGS["olmoe-1b-7b"], "allenai/OLMoE-1B-7B-0924", 16,
                               (("hidden_size", "hidden_size"), ("intermediate_size", "intermediate_size"),
                                ("num_experts", "num_experts"), ("num_experts_per_tok", "top_k"),
                                ("num_hidden_layers", "num_layers"))),
    "qwen3-30b-a3b": ServedModel("qwen3-30b-a3b", QWEN3, "Qwen/Qwen3-30B-A3B", 48,
                                 (("hidden_size", "hidden_size"), ("moe_intermediate_size", "intermediate_size"),
                                  ("num_experts", "num_experts"), ("num_experts_per_tok", "top_k"),
                                  ("num_hidden_layers", "num_layers")),
                                 (("decoder_sparse_step", 1), ("mlp_only_layers", []))),
}

# --------------------------------------------------------------------------------------------------------------
# THE BLIND-MODEL REGISTRY (2026-11-20): one dict entry per new model, read by prodlib (MODELS, HF_REVISIONS),
# frontier_op (FRONTIER_MODELS, PROBE_POINTS) and main.sh (--model KEY: TOK, levels, probe points)
# --------------------------------------------------------------------------------------------------------------

#: The config.json keys G-CFG holds a blind model to when its entry names none (the Qwen / Mixtral dialect)
DEFAULT_HF_KEYS = {"hidden_size": "hidden_size", "moe_intermediate_size": "intermediate_size",
                   "num_experts": "num_experts", "num_experts_per_tok": "top_k", "num_hidden_layers": "num_layers"}
DEFAULT_PROBE_POINTS = (64, 1024, 8192)

#: {model key: spec}. Required: hf_repo, hidden_size, moe_intermediate_size, num_experts, top_k, num_layers. Optional:
#: first_moe_layer (0), moe_layers (num_layers - first_moe_layer), shared_experts (0; a separate dense MLP is not in
#: the measured call), gate_fn ("softmax"), norm_topk_prob (True), hf_keys ({config.json key: MoEConfig attribute},
#: DEFAULT_HF_KEYS), hf_literals ({config.json key: value}), levels (serving levels; default main.sh's),
#: probe_points (DEFAULT_PROBE_POINTS), frontier_name (f"{key}-moe"), frontier_config (Frontier's HF-config dialect
#: beyond the MoE numbers: attention heads, vocab, ...; cosmetic for the explicit-parameter trainer). Every number is
#: checked against the downloaded config.json by G-CFG before the first step. Example (NOT a registered model):
#:     "acme-moe-16b-a2b": {"hf_repo": "acme/MoE-16B-A2B", "hidden_size": 2048, "moe_intermediate_size": 1024,
#:                          "num_experts": 64, "top_k": 6, "num_layers": 28, "first_moe_layer": 1},
#: The model's HF commit goes in BLIND_REVISIONS_PENDING, never here.
BLIND_SPECS: dict = {
    # 2026-11-20, the v9 / v8 / Frontier blind run: three models neither predictor has seen (no calibration or test
    # page, not in Frontier's paper), each through vLLM 0.27.1's Triton fused_moe in bf16 (no GH200 tuned config:
    # the default tile ladder). Every number is G-CFG-checked against config.json at the pinned revision.
    "glm-4.7-flash": {"hf_repo": "zai-org/GLM-4.7-Flash", "hidden_size": 2048, "moe_intermediate_size": 1536,
                      "num_experts": 64, "top_k": 4, "num_layers": 47, "first_moe_layer": 1, "shared_experts": 1,
                      "gate_fn": "sigmoid", "norm_topk_prob": True,
                      "hf_keys": {"hidden_size": "hidden_size", "moe_intermediate_size": "intermediate_size",
                                  "n_routed_experts": "num_experts", "num_experts_per_tok": "top_k",
                                  "num_hidden_layers": "num_layers"},
                      "levels": "1,8,24,48,80,112,256"},
    "trinity-mini": {"hf_repo": "arcee-ai/Trinity-Mini", "hidden_size": 2048, "moe_intermediate_size": 1024,
                     "num_experts": 128, "top_k": 8, "num_layers": 32, "first_moe_layer": 2, "shared_experts": 1,
                     "gate_fn": "sigmoid", "norm_topk_prob": True, "levels": "1,8,24,48,80,112,256"},
    "ling-mini-2.0": {"hf_repo": "inclusionAI/Ling-mini-2.0", "hidden_size": 2048, "moe_intermediate_size": 512,
                      "num_experts": 256, "top_k": 8, "num_layers": 20, "first_moe_layer": 1, "shared_experts": 1,
                      "gate_fn": "sigmoid", "norm_topk_prob": True, "levels": "1,8,24,48,80,112,256"},
}

#: PLACEHOLDER: {model key: the HF commit sha of the model's repo}, to be FILLED by a separate step from
#: `python scripts/prod/hf_revisions.py` (the Hub's public API) once the candidate models are verified. A key of
#: BLIND_SPECS missing here (or None) is UNPINNED: main.sh stops at revision-pinned and serve_capture refuses it.
BLIND_REVISIONS_PENDING: dict = {   # HF main commits read from /api/models on 2026-11-20
    "glm-4.7-flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "trinity-mini": "e7fa016763d03089ab547c9ee19044980073669b",
    "ling-mini-2.0": "a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab",
}

_SPEC_REQUIRED = ("hf_repo", "hidden_size", "moe_intermediate_size", "num_experts", "top_k", "num_layers")


def served_from_spec(key: str, s: dict) -> ServedModel:
    """The ServedModel (and its MoEConfig) a BLIND_SPECS entry describes; KitError on a missing required field."""
    miss = [f for f in _SPEC_REQUIRED if f not in s]
    if miss:
        raise KitError(f"BLIND_SPECS[{key!r}]: missing {miss}")
    first = int(s.get("first_moe_layer", 0))
    cfg = MoEConfig(name=key, hidden_size=int(s["hidden_size"]), intermediate_size=int(s["moe_intermediate_size"]),
                    num_experts=int(s["num_experts"]), top_k=int(s["top_k"]), gate_fn=s.get("gate_fn", "softmax"),
                    norm_topk_prob=bool(s.get("norm_topk_prob", True)), shared_experts=int(s.get("shared_experts", 0)),
                    num_layers=int(s["num_layers"]), first_moe_layer=first, hf_repo=s["hf_repo"], verified=True)
    hk = s.get("hf_keys") or DEFAULT_HF_KEYS
    lit = s.get("hf_literals") or {}
    return ServedModel(key, cfg, s["hf_repo"], int(s.get("moe_layers", int(s["num_layers"]) - first)),
                       tuple(hk.items()), tuple(lit.items()))


def register_specs(specs: dict, revisions: dict) -> list[str]:
    """Add every spec to MODELS and its pin (or None: unpinned) to HF_REVISIONS; a key already in MODELS is refused
    (a registered model is never redefined). Returns the keys added."""
    for key, s in specs.items():
        if key in MODELS:
            raise KitError(f"BLIND_SPECS[{key!r}]: already a served model")
        MODELS[key] = served_from_spec(key, s)
        HF_REVISIONS[key] = revisions.get(key)
    return list(specs)


def spec_of(key: str) -> dict:
    """The model's BLIND_SPECS entry ({} for the kit's own models)."""
    return dict(BLIND_SPECS.get(key) or {})


def frontier_name(key: str) -> str:
    """Frontier's model name for the key (frontier_op.FRONTIER_MODELS; a blind model's frontier_name or <key>-moe)."""
    own = {"ernie-4.5-21b-a3b": "ernie45-21b-a3b-moe", "olmoe-1b-7b": "olmoe-1b-7b-moe",
           "qwen3-30b-a3b": "qwen3-30b-a3b-moe"}
    return own.get(key) or spec_of(key).get("frontier_name") or f"{key}-moe"


def frontier_csv_rel(key: str) -> str:
    """frontier_op.csv_relpath(key) without importing it (a test holds the two equal)."""
    return f"compute/gh200/{frontier_name(key)}/moe_kernel_only.csv"


def run_names(key: str, utc: str) -> dict:
    """Per-instance names (one blind model per GH200): the results dir prod-<key>-<utc> and the run branch
    run-gh200-blind-<key>-<utc> (lower case: vm_run.sh's branch rule ^run-[a-z0-9][a-z0-9.-]*$)."""
    if key not in MODELS:
        raise KitError(f"unknown model {key!r}")
    if not utc or not all(c.isalnum() for c in utc):
        raise KitError(f"utc stamp {utc!r}: want e.g. 20261120T035826Z")
    return {"results": f"prod-{key}-{utc}", "branch": f"run-gh200-blind-{key}-{utc}".lower()}


def expert_bytes(cfg: MoEConfig) -> int:
    """One expert's bf16 weight bytes, w13 (2 I x H) and w2 (H x I): the replay budget's scale between models (review 2,
    MAJOR B): OLMoE-1B-7B 12.6 MB, ERNIE-4.5-21B-A3B 23.6 MB."""
    return 3 * int(cfg.hidden_size) * int(cfg.intermediate_size) * 2


def hf_config_view(config) -> dict:
    """The dict G-CFG reads: config.to_dict() plus every alias in the config's attribute_map whose key to_dict()
    dropped. transformers 5.x stores Qwen3-MoE's expert count as num_local_experts and keeps num_experts only as an
    attribute_map alias, so to_dict() has no num_experts although config.json does (the 2026-11-20 extras X3 G-CFG
    stop). An alias fills a missing key only; a key to_dict() already holds is never replaced."""
    d = dict(config.to_dict())
    for alias, canonical in (getattr(config, "attribute_map", None) or {}).items():
        if alias not in d and canonical in d:
            d[alias] = d[canonical]
    return d


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
    for key, want in model.hf_literals:
        got = hf.get(key)
        if got != want:
            bad.append(f"config.json {key}={got!r}, the kit requires {want!r}")
    return bad


# --------------------------------------------------------------------------------------------------------------
# tiles, branches, strata
# --------------------------------------------------------------------------------------------------------------

DECODE_MAX_T = 128
#: replay.py --cell-marks: the cell index rides in the step marker's gridY (moe_capture.STEP_MOD, CUDA's limit)
CELL_MARK_MAX = 65535
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


def tile_bins(E: int = 64, max_T: int | None = None) -> tuple:
    """The seven default tiles in T order at E experts: the BLOCK_M / BLOCK_N / BLOCK_K ladder depends on T only, and
    GROUP_SIZE_M turns 16 once T // E > 128, i.e. T >= 129 E (E 64: 8256; E 128: 16512). With max_T, only the tiles a
    step of at most max_T tokens can reach (serve_capture's coverage gate: Qwen3's G16 starts past the registered
    16384-token budget)."""
    g16 = 129 * int(E)
    bins = (("16x64x128w4s4", 1, 32), ("32x64x128w4s3", 33, 64), ("32x128x64w4s3", 65, 96),
            ("64x128x64w4s3", 97, 128), ("64x128x64w8s3", 129, 512), ("128x128x64w8s3", 513, g16 - 1),
            ("128x128x64w8s3 g16", g16, 10 ** 9))
    return bins if max_T is None else tuple(b for b in bins if b[1] <= int(max_T))


def branch_tiles(E: int, branch_name: str, max_T: int | None = None) -> tuple:
    """The tile labels of one branch (decode: T <= DECODE_MAX_T, prefill: above), of tile_bins(E, max_T). G-COV
    per run (2026-11-20): the 1710 run is gated on the decode tiles only, the 1410 run on the prefill tiles only
    (the A1710 capture stopped on the prefill G16 tile, which the 1410 run is the one to measure)."""
    if branch_name not in ("decode", "prefill", "all"):
        raise ValueError(f"branch {branch_name!r}")
    return tuple(lab for lab, lo, _hi in tile_bins(E, max_T)
                 if branch_name == "all" or branch(lo) == branch_name)


#: the seven tiles a serving run at E=64 lands on, in T order (KIT-SPEC 2 / CRITIQUE M9)
TILE_BINS = tile_bins(64)


def tile_bin(T: int, E: int = 64) -> str:
    for lab, lo, hi in (TILE_BINS if int(E) == 64 else tile_bins(E)):
        if lo <= T <= hi:
            return lab
    raise ValueError(f"T {T} is not a token count")


def branch(T: int) -> str:
    return "decode" if int(T) <= DECODE_MAX_T else "prefill"


def _refused_bounds(top_k: int, E: int) -> tuple[int, int]:
    return max(-(-HOPPER_G1_N * int(E) * 128 // int(top_k)), 513), 129 * int(E) - 1


def refused_stratum(top_k: int, E: int = 64) -> str:
    """The refused window's stratum name, PER MODEL (review of the extras, MINOR): its own bounds in the name, so a
    Qwen3 call (E 128, k 8: 12288..16511) is never reported under ERNIE's 8192..8255."""
    lo, hi = _refused_bounds(top_k, E)
    return f"refused_window_{lo}_{hi}"


#: ERNIE's refused-window stratum (the registered window, E 64, k 6); every model's own is refused_stratum(k, E)
REFUSED_STRATUM = refused_stratum(ERNIE.top_k, ERNIE.num_experts)
assert REFUSED_STRATUM == "refused_window_8192_8255"


def strata(T: int, top_k: int, E: int = 64) -> list[str]:
    """The strata a call belongs to beyond its tile (CRITIQUE M4, M5). The refused window is the model's own
    (refused_window, k-dependent), named by its own bounds (refused_stratum)."""
    out = []
    T = int(T)
    if 129 <= T <= 512:
        out.append("band_64x128")
    if T > DECODE_MAX_T and T * top_k < SMALL_BATCH_NUMEL and E <= 64:
        out.append("small_batch_align_on_prefill_tile")
    lo, hi = _refused_bounds(top_k, E)
    if lo <= T <= hi:
        out.append(refused_stratum(top_k, E))
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
# the per-call outputs, sharded below the push cap (review 2, MAJOR A)
# --------------------------------------------------------------------------------------------------------------

#: scripts/vm_results_push.sh (registered, pinned: not edited) xz-compresses or HOLDS BACK on the VM any file over
#: 95 MB. Every per-call output the laptop needs is written in shards of at most SHARD_MAX_BYTES (a third below the
#: cap), so a push carries all of it as plain files; main.sh's last hard step (held-back-check) fails if the push
#: still held back a needed path.
SHARD_MAX_BYTES = 64_000_000
SHARDS_SCHEMA = "moe-kernels/prod-call-shards/1"


def _shard_name(stem: str, ext: str, i: int, n: int) -> str:
    return f"{stem}{ext}" if n == 1 else f"{stem}.{i:05d}{ext}"


def write_calls(out_dir, rows: list[dict], counts: np.ndarray, *, prefix: str = "",
                max_bytes: int = SHARD_MAX_BYTES) -> dict:
    """<prefix>calls.jsonl and <prefix>counts.npy, each split into shards of at most max_bytes (one shard keeps the
    plain name; more are <stem>.00000<ext>, ...), and <prefix>calls.shards.json listing every shard with its row count
    and sha256 (read_calls / load_counts refuse a missing or changed shard). Returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts)
    if len(rows) != len(counts):
        raise KitError(f"{len(rows)} calls, {len(counts)} count rows")
    for old in list(out.glob(f"{prefix}calls.[0-9]*.jsonl")) + list(out.glob(f"{prefix}counts.[0-9]*.npy")):
        old.unlink()
    chunks: list[list[bytes]] = [[]]
    size = 0
    for r in rows:
        line = (json.dumps(r, default=str) + "\n").encode()
        if chunks[-1] and size + len(line) > max_bytes:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line)
    man: dict = {"schema": SHARDS_SCHEMA, "n": len(rows), "max_bytes": int(max_bytes), "calls": [], "counts": []}
    for i, ch in enumerate(chunks):
        name = _shard_name(f"{prefix}calls", ".jsonl", i, len(chunks))
        data = b"".join(ch)
        (out / name).write_bytes(data)
        man["calls"].append({"file": name, "rows": len(ch), "sha256": hashlib.sha256(data).hexdigest()})
    row_bytes = max(1, int(counts.shape[1]) * counts.dtype.itemsize) if counts.ndim == 2 else 1
    per = max(1, (max_bytes - 1024) // row_bytes)
    bounds = list(range(0, len(counts), per)) or [0]
    for i, lo in enumerate(bounds):
        name = _shard_name(f"{prefix}counts", ".npy", i, len(bounds))
        part = counts[lo:lo + per]
        np.save(out / name, part)
        man["counts"].append({"file": name, "rows": int(len(part)),
                              "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()})
    (out / f"{prefix}calls.shards.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def _read_bytes(p: Path) -> bytes:
    """A file's bytes, or its .xz twin's decompressed bytes (vm_results_push.sh's compressed form)."""
    if p.exists():
        return p.read_bytes()
    xz = p.with_name(p.name + ".xz")
    if xz.exists():
        import lzma
        return lzma.decompress(xz.read_bytes())
    raise KitError(f"{p}: missing (and no {xz.name})")


def _shards(d: Path, prefix: str, kind: str) -> list[tuple[bytes, int | None]]:
    man_p = d / f"{prefix}calls.shards.json"
    if not man_p.exists():
        plain = d / (f"{prefix}calls.jsonl" if kind == "calls" else f"{prefix}counts.npy")
        if not plain.exists() and not plain.with_name(plain.name + ".xz").exists():
            if kind == "calls" and list(d.glob(f"{prefix}calls.[0-9]*.jsonl*")):
                raise KitError(f"{d}: call shards without their manifest {man_p.name}")
            return []
        return [(_read_bytes(plain), None)]
    man = json.loads(man_p.read_text())
    if man.get("schema") != SHARDS_SCHEMA:
        raise KitError(f"{man_p}: schema {man.get('schema')!r}")
    out = []
    for ent in man[kind]:
        data = _read_bytes(d / ent["file"])
        if hashlib.sha256(data).hexdigest() != ent["sha256"]:
            raise KitError(f"{d / ent['file']}: sha256 differs from {man_p.name}")
        out.append((data, int(ent["rows"])))
    return out


def read_calls(run_dir, prefix: str = "") -> list[dict]:
    """Every call row of a run, across its shards (or the plain / .xz single file). [] when the run has none."""
    rows: list[dict] = []
    for data, n in _shards(Path(run_dir), prefix, "calls"):
        part = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if n is not None and len(part) != n:
            raise KitError(f"{run_dir}: a calls shard holds {len(part)} rows, the manifest says {n}")
        rows += part
    return rows


def load_counts(run_dir, prefix: str = "") -> np.ndarray:
    """The [calls, E] counts of a run, across its shards (or the plain / .xz single file)."""
    import io
    parts = []
    for data, n in _shards(Path(run_dir), prefix, "counts"):
        a = np.load(io.BytesIO(data), allow_pickle=False)
        if n is not None and len(a) != n:
            raise KitError(f"{run_dir}: a counts shard holds {len(a)} rows, the manifest says {n}")
        parts.append(a)
    if not parts:
        raise KitError(f"{run_dir}: no counts.npy (or shards)")
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def has_calls(run_dir, prefix: str = "") -> bool:
    d = Path(run_dir)
    return any((d / n).exists() for n in (f"{prefix}calls.shards.json", f"{prefix}calls.jsonl",
                                          f"{prefix}calls.jsonl.xz"))


#: what a push may hold back without losing anything the laptop needs (the rental's own intermediates): the nsys
#: reports and their sqlite exports (extracted on the VM), the downloaded dataset (pinned by sha256, re-fetchable),
#: the replay's Triton caches
PUSH_EXPENDABLE = ("*.nsys-rep", "*.qdstrm", "*.sqlite", "data/*", "*triton-cache/*")


def held_back_needed(held_text: str, run_name: str) -> list[str]:
    """The lines of vm_results_push.sh's HELD-BACK.txt (path TAB size TAB sha TAB fate) that hold back a NEEDED file
    of the run directory `run_name` (anything under it but PUSH_EXPENDABLE), compressed or left on the VM."""
    import fnmatch
    bad = []
    key = f"/{run_name}/"
    for line in held_text.splitlines():
        rel = line.split("\t", 1)[0].strip()
        if not rel or key not in f"/{rel}":
            continue
        sub = f"/{rel}".split(key, 1)[1]
        if not any(fnmatch.fnmatch(sub, pat) for pat in PUSH_EXPENDABLE):
            bad.append(line.strip())
    return bad


#: main.sh writes every raw trace (*.nsys-rep, *.sqlite) under $MOE_HOME/traces/<run>/, OUTSIDE the pushed results
#: tree (review 2026-11-19 m6: vm_results_push.sh, pinned, would xz-compress multi-GB traces on every push); kit_push
#: records their sha256 here, in the results tree, and scripts/prod/pull_traces.sh copies and checks them
TRACES_MANIFEST = "traces.sha256.json"
TRACES_SCHEMA = "moe-kernels/prod-traces/1"


def trace_manifest(traces, out) -> dict:
    """sha256 of every file under `traces` written to `out` (JSON); a file whose size and mtime are unchanged since
    the last manifest keeps its hash (a push re-hashes only the new traces)."""
    root, out = Path(traces), Path(out)
    old = {}
    if out.is_file():
        try:
            old = json.loads(out.read_text()).get("files", {})
        except (ValueError, OSError):
            old = {}
    files = {}
    if root.is_dir():
        for f in sorted(x for x in root.rglob("*") if x.is_file()):
            rel = str(f.relative_to(root))
            st = f.stat()
            prev = old.get(rel)
            if prev and prev.get("bytes") == st.st_size and prev.get("mtime_ns") == st.st_mtime_ns:
                files[rel] = prev
                continue
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for buf in iter(lambda: fh.read(1 << 24), b""):
                    h.update(buf)
            files[rel] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()}
    doc = {"schema": TRACES_SCHEMA, "traces_root": str(root), "run": root.name, "files": files,
           "total_bytes": sum(v["bytes"] for v in files.values())}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.rename(out)
    return doc


#: the runs main.sh measures; the primaries are hard steps there, so the laptop always expects them
PRIMARY_RUNS = ("A1710", "A1410")
SECONDARY_RUNS = ("Adefault", "Aon", "Beager")
#: main.sh's Frontier steps whose PASS means a file the laptop reads (frontier_op.py: its predictor, our profiles)
#: (each step tied to the file it really writes: review m8; main.sh is ERNIE's, frontier_op.csv_relpath)
FRONTIER_CSV_REL = "compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"
FRONTIER_OUTPUTS = (("frontier-predict", "frontier/predictions/index.json"),
                    ("frontier-points", "frontier/points.json"),
                    ("frontier-csv", f"frontier/profile/{FRONTIER_CSV_REL}"),
                    ("frontier-csv-fills", f"frontier/profile-fills/{FRONTIER_CSV_REL}"),
                    ("frontier-csv-partial", "frontier/profile-partial/csv_meta.json"),
                    ("frontier-holdout", "frontier/holdout/holdout.json"),
                    ("frontier-timer-probe", "frontier-probe/predictions/index.json"))


#: main.sh writes the run's profile and model here (blind: no secondary runs, no replays); absent = the full profile
RUN_PROFILE_FILE = "kit-profile.json"
PROFILES = ("full", "blind")


def run_profile(results) -> dict:
    """{"profile": "full" | "blind", "model": key} from <results>/kit-profile.json (absent: full, ERNIE)."""
    p = Path(results) / RUN_PROFILE_FILE
    doc = json.loads(p.read_text()) if p.is_file() else {}
    prof = doc.get("profile", "full")
    if prof not in PROFILES:
        raise KitError(f"{p}: profile {prof!r}, not one of {PROFILES}")
    return {"profile": prof, "model": doc.get("model", "ernie-4.5-21b-a3b")}


def expected_outputs(results, profile: str | None = None) -> dict:
    """What offline.sh must find in a fetched results tree, from the rental's own SUMMARY.txt: the primary runs,
    every secondary run whose extract step PASSED, the board par, both replays, each replay_check.json whose check
    PASSED, and each Frontier output whose step PASSED (FRONTIER_OUTPUTS). {"runs": [labels], "frontier": [paths],
    "missing": [why]}. Under the BLIND profile (main.sh --profile blind: kit-profile.json, or `profile`) the
    secondary runs and the replays were never planned and are not required; the Frontier CSV is the run's model's."""
    R = Path(results)
    summ = R / "SUMMARY.txt"
    if not summ.exists():
        return {"runs": [], "frontier": [], "missing": [f"{summ}: no SUMMARY.txt (cannot tell what the rental measured)"]}
    rp = run_profile(R)
    blind = (profile or rp["profile"]) == "blind"
    passed = {ln.split()[1] for ln in summ.read_text().splitlines() if ln.startswith("PASS ") and len(ln.split()) > 1}
    runs = list(PRIMARY_RUNS) + ([] if blind else [lab for lab in SECONDARY_RUNS if f"{lab}-extract" in passed])
    missing = [f"run {lab}: no calls under {R / lab}" for lab in runs if not has_calls(R / lab)]
    fr = []
    csv_rel = frontier_csv_rel(rp["model"])
    for stepname, rel in FRONTIER_OUTPUTS:
        rel = rel.replace(FRONTIER_CSV_REL, csv_rel)
        if stepname in passed:
            fr.append(rel)
            if not (R / rel).exists() and not (R / (rel + ".xz")).exists():
                missing.append(f"{rel}: {stepname} PASSED on the rental, the file is missing")
    for b in ("decode", "prefill"):
        rel = f"replay-offset-{b}/replay_check.json"
        if f"replay-offset-check-{b}" in passed and not (R / rel).exists() and not (R / (rel + ".xz")).exists():
            missing.append(f"{rel}: replay-offset-check-{b} PASSED on the rental, the file is missing")
    if not (R / "sag" / "par.json").exists() and not (R / "sag" / "par.json.xz").exists():
        missing.append(f"{R / 'sag' / 'par.json'}: missing")
    for b in ("decode", "prefill"):
        if not blind and not any((R / f"replay-{b}").rglob("cells.csv*")):
            missing.append(f"replay-{b}: no cells.csv under {R / f'replay-{b}'}")
    return {"runs": runs, "frontier": fr, "missing": missing}


def decompress_tree(root) -> list[str]:
    """Every <f>.xz under root (vm_results_push.sh's compressed form) written out as <f> beside it when <f> is absent."""
    import lzma
    done = []
    for xz in sorted(Path(root).rglob("*.xz")):
        plain = xz.with_name(xz.name[:-3])
        if plain.exists():
            continue
        tmp = plain.with_name(plain.name + ".part")
        with lzma.open(xz, "rb") as src, open(tmp, "wb") as dst:
            while True:
                buf = src.read(1 << 24)
                if not buf:
                    break
                dst.write(buf)
        tmp.rename(plain)
        done.append(str(plain))
    return done


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

#: {model key: commit sha of the HF repo}. None = NOT PINNED (main.sh refuses an unpinned model at its first step,
#: revision-pinned; serve_capture's G-REV FAILS a pinned run whose loaded snapshot differs or cannot be resolved). The same sha pins the tokenizer (prompts.py) and the engine's
#: model and tokenizer (serve_capture.py: LLM(revision=, tokenizer_revision=)).
#: Resolved on the laptop 2026-11-19 by scripts/prod/hf_revisions.py (the Hub's public model API, no auth; review 2,
#: MINOR 3) and pasted here; `hf_revisions.py --check` holds them to the Hub before the kit freezes.
HF_REVISIONS: dict = {"ernie-4.5-21b-a3b": "87db95487941cb39592ee0abca3b9155a6d19c5c",
                      "olmoe-1b-7b": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
                      # extras.sh X3: the Hub's sha of Qwen/Qwen3-30B-A3B on 2026-11-19 (public model API, no auth)
                      "qwen3-30b-a3b": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}


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


def predicted_reads(ans: dict) -> dict | None:
    """v8's predicted DRAM reads of one priced call, {"w1": bytes, "w2": bytes, "source": where}, or None when v8's
    answer carries none (a refusal, or a decode path whose parts hold no reads: NOT AVAILABLE). Prefill: price()'s
    own `reads` (round 5's per-GEMM reads, RR's w1 when RR applies); decode: the v6d cell's `parts.reads`."""
    if not isinstance(ans, dict) or ans.get("ms") is None:
        return None
    for src, rd in (("reads", ans.get("reads")), ("parts.reads", (ans.get("parts") or {}).get("reads"))):
        if isinstance(rd, dict) and rd.get("w1") is not None and rd.get("w2") is not None:
            return {"w1": float(rd["w1"]), "w2": float(rd["w2"]), "source": src}
    return None


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


def shell_model_info(key: str) -> str:
    """main.sh --model KEY: TOK (the model's own repo: prompts.py pins the tokenizer by the model's revision), LEVELS
    (the spec's serving levels, empty = main.sh's default), PROBE_POINTS, CSVREL (Frontier's CSV under profile/)."""
    import shlex
    m = MODELS[key]
    s = spec_of(key)
    pts = s.get("probe_points") or {"olmoe-1b-7b": (64, 192, 256)}.get(key, DEFAULT_PROBE_POINTS)
    kv = {"TOK": m.hf_repo, "SPEC_LEVELS": s.get("levels", ""), "SPEC_PROBE_POINTS": ",".join(str(int(x)) for x in pts),
          "CSVREL": frontier_csv_rel(key)}
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in kv.items())


def prefetch(key: str) -> int:
    """The model's config, tokenizer and weights at its pin into the HF cache (huggingface_hub.snapshot_download),
    so the gate's first engine load does not download. Refuses an unpinned model."""
    m = MODELS[key]
    rev = hf_revision(key)
    if not rev:
        raise KitError(f"{key} is not pinned (HF_REVISIONS / BLIND_REVISIONS_PENDING)")
    from huggingface_hub import snapshot_download
    path = snapshot_download(m.hf_repo, revision=rev,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py"])
    print(f"PASS prefetch {key}: {m.hf_repo} at {rev} -> {path}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="prodlib: the kit's v8 contract check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("price-check")
    pc.add_argument("--model", required=True, choices=sorted(MODELS))
    pc.add_argument("--pricer", default=None)
    pc.add_argument("--sag", default=None, help="the board's par.json (default: the registered reference median)")
    hb = sub.add_parser("held-back-check", help="FAIL if the push held back a needed file of this run (main.sh)")
    hb.add_argument("--held", required=True, help="vm_results_push.sh's HELD-BACK.txt ($MOE_PUSH_HOME/push/)")
    hb.add_argument("--results", required=True, help="the run's results directory")
    ex = sub.add_parser("expected-outputs", help="FAIL if a fetched tree lacks what the rental measured (offline.sh)")
    ex.add_argument("--results", required=True)
    ex.add_argument("--runs-out", required=True, help="writes the labels of the runs to score, one per line")
    ex.add_argument("--profile", choices=PROFILES, default=None, help="default: <results>/kit-profile.json, else full")
    mi = sub.add_parser("model-info", help="main.sh --model KEY: shell assignments TOK, LEVELS, PROBE_POINTS, CSVREL")
    mi.add_argument("--model", required=True)
    rn = sub.add_parser("run-names", help="the per-instance results dir and run branch (launcher, main.sh)")
    rn.add_argument("--model", required=True)
    rn.add_argument("--utc", required=True)
    pf = sub.add_parser("prefetch", help="download the model's files at its pin into the HF cache (main.sh, bg)")
    pf.add_argument("--model", required=True)
    dz = sub.add_parser("decompress", help="write every <f>.xz under a tree out as <f> (offline.sh)")
    dz.add_argument("--root", required=True)
    tm = sub.add_parser("trace-manifest", help="sha256 of the run's raw traces into the results tree (kit_push)")
    tm.add_argument("--traces", required=True, help="$MOE_HOME/traces/<run>")
    tm.add_argument("--out", required=True, help=f"<results>/{TRACES_MANIFEST}")
    a = ap.parse_args(argv)
    if a.cmd == "trace-manifest":
        doc = trace_manifest(a.traces, a.out)
        print(f"PASS trace-manifest: {len(doc['files'])} file(s), {doc['total_bytes'] / 1e9:.2f} GB under {a.traces} "
              f"-> {a.out}")
        return 0
    if a.cmd == "held-back-check":
        held = Path(a.held)
        if not held.exists():
            print(f"FAIL held-back-check: no {held} (the results were never pushed)")
            return 3
        run_name = Path(a.results).resolve().name
        sums = held.with_name("SHA256SUMS")
        if not sums.exists() or f"/{run_name}/" not in sums.read_text():
            print(f"FAIL held-back-check: the push's manifest {sums} holds no file of {run_name} (is the results "
                  "directory under the pushed ~/moe/results?)")
            return 3
        bad = held_back_needed(held.read_text(), run_name)
        if bad:
            print(f"FAIL held-back-check: the push held back {len(bad)} needed file(s) of {a.results} "
                  f"(copy them off the VM before the terminate): " + " | ".join(bad[:6]))
            return 3
        print(f"PASS held-back-check: no needed file of {Path(a.results).name} held back")
        return 0
    if a.cmd in ("model-info", "run-names", "prefetch"):
        import shlex
        try:
            if a.model not in MODELS:
                raise KitError(f"unknown model {a.model!r} (prodlib.MODELS: {sorted(MODELS)})")
            if a.cmd == "run-names":
                n = run_names(a.model, a.utc)
                print(f"RESULTS_NAME={shlex.quote(n['results'])}\nRUN_BRANCH={shlex.quote(n['branch'])}")
            elif a.cmd == "model-info":
                print(shell_model_info(a.model))
            else:
                return prefetch(a.model)
        except KitError as exc:
            print(f"FAIL {a.cmd}: {exc}")
            return 3
        return 0
    if a.cmd == "expected-outputs":
        got = expected_outputs(a.results, a.profile)
        if got["missing"]:
            print("FAIL expected-outputs: " + "; ".join(got["missing"]))
            return 3
        Path(a.runs_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.runs_out).write_text("".join(f"{r}\n" for r in got["runs"]))
        print(f"PASS expected-outputs: runs {got['runs']}, Frontier {got['frontier']}")
        return 0
    if a.cmd == "decompress":
        done = decompress_tree(a.root)
        print(f"PASS decompress: {len(done)} file(s) written from .xz" + (f": {done[:6]}" if done else ""))
        return 0
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


register_specs(BLIND_SPECS, BLIND_REVISIONS_PENDING)

if __name__ == "__main__":
    sys.exit(main())
