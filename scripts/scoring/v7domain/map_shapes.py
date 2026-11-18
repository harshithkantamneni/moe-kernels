"""The v7 domain map's 14 SYNTHETIC decode shapes (registration docs/registered/2026-11-18-v7-domain-map-gh200), named
map-<family>-e<E>k<k>-h<H>i<I>: no real model's name, config, checkpoint or hf_repo; synthetic weights, as the apparatus
always uses. Phase 0's map design (moe-kernels-work 2026-11-18 domain-phase0, p6_map.py), unchanged in geometry:

  A  path twins across E at one slab (H 2048, I 512: W_e 6.29 MB), (E, k) = (8,2) (16,4) (32,8) (64,8) (128,8) (256,8)
  B  slab sweep at E 32, k 4: (H, I) = (1024,384) (1536,512) (2048,768) (2560,1152) (3072,1536) (4096,1536)
  C  block-scan at E 256, k 8: (H, I) = (7168,384) (4096,1024)

Every other field is Phase 0's (granite-3.0-1b-a400m's MoEConfig with the geometry replaced, hf_repo None, verified
True, as the toy config): softmax gate, no renormalisation, no shared expert, 24 layers. Routing never reads them: each
decode cell names its counts.

WHY NOT IN moe/spec.py. moe/spec.py is frozen at sha256 58fb9b7f... by twelve registrations (CAL2, CAL4, tileB3, decode
addenda 2 to 6, PC, PC2, RR and the v7 gate; the v7 decode blind through the gate's freeze); adding 14 entries there
voids each of those freezes (16 tests fail; gate_v7.py and score_blind_v7.py refuse). So the shapes live here, and reach moe.spec.MODEL_CONFIGS:
  - on the laptop (registration, readout, tests): through `installed()`, which adds them for one block and removes them;
  - on the VM: through scripts/scoring/v7domain/vmsite/sitecustomize.py, which scripts/vm_run_v7m.sh puts on the
    driver's PYTHONPATH; it adds them when a process first imports moe.spec, so the pinned decode driver, r3_decode.py
    and locked_r3.py run unchanged.
This module imports nothing from the repository at import time (the VM hook loads it by path)."""
from __future__ import annotations

import contextlib
import dataclasses

#: (family, E, k, H, I) in Phase 0's order
GEOMETRY = (
    ("A", 8, 2, 2048, 512), ("A", 16, 4, 2048, 512), ("A", 32, 8, 2048, 512), ("A", 64, 8, 2048, 512),
    ("A", 128, 8, 2048, 512), ("A", 256, 8, 2048, 512),
    ("B", 32, 4, 1024, 384), ("B", 32, 4, 1536, 512), ("B", 32, 4, 2048, 768), ("B", 32, 4, 2560, 1152),
    ("B", 32, 4, 3072, 1536), ("B", 32, 4, 4096, 1536),
    ("C", 256, 8, 7168, 384), ("C", 256, 8, 4096, 1024),
)
#: the fields Phase 0 carried from granite-3.0-1b-a400m (routing never reads them)
FIXED = {"gate_fn": "softmax", "norm_topk_prob": False, "routed_scaling_factor": 1.0, "shared_experts": 0,
         "num_layers": 24, "first_moe_layer": 0, "tensor_parallel": 1, "hf_repo": None, "verified": True}
PREFIX = "map-"


def name_of(fam: str, E: int, k: int, H: int, I: int) -> str:  # noqa: E741
    return f"{PREFIX}{fam.lower()}-e{E}k{k}-h{H}i{I}"


SHAPES = tuple((name_of(*g), *g) for g in GEOMETRY)
NAMES = tuple(s[0] for s in SHAPES)
FAMILY = {s[0]: s[1] for s in SHAPES}
#: Phase 0's working names (p6_map.py, map.json), for the record and the reproduction check
PHASE0_NAME = {name_of(f, E, k, H, I): (f"syn{f}-E{E}k{k}" if f == "A" else f"syn{f}-H{H}I{I}")
               for f, E, k, H, I in GEOMETRY}


def configs(spec) -> dict:
    """{name: MoEConfig} built with `spec` (the moe.spec module)."""
    return {n: spec.MoEConfig(name=n, hidden_size=H, intermediate_size=I, num_experts=E, top_k=k, **FIXED)
            for n, _f, E, k, H, I in SHAPES}


def install(spec) -> list[str]:
    """Adds the 14 shapes to spec.MODEL_CONFIGS; refuses a name already there with another config. Returns the names
    it added (none when already installed)."""
    want = configs(spec)
    added = []
    for n, c in want.items():
        have = spec.MODEL_CONFIGS.get(n)
        if have is None:
            spec.MODEL_CONFIGS[n] = c
            added.append(n)
        elif have != c:
            raise ValueError(f"{n}: moe.spec already holds another config under this name")
    return added


@contextlib.contextmanager
def installed():
    """The 14 shapes in moe.spec.MODEL_CONFIGS for one block (laptop side), removed after."""
    import moe.spec as S
    added = install(S)
    try:
        yield
    finally:
        for n in added:
            S.MODEL_CONFIGS.pop(n, None)


def as_dicts() -> list[dict]:
    return [{"name": n, "family": f, "E": E, "k": k, "H": H, "I": I, **FIXED} for n, f, E, k, H, I in SHAPES]


def fields_of(cfg) -> dict:
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
