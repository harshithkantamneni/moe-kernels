#!/usr/bin/env python
"""G=1 slab survival off one r3c page, offline (rental 2, docs/registered
rental2-knobs). Lifted from scripts/scoring/rental1/score_l2.surv, with every
term written down so the registration states the formula and not a script:

  s(n)    = (B_PRIVATE(n) - B_SHARED(n)) / ((n - 1) W_g), n = 2..9, B the cell's
            per-GEMM `dram_bytes_read` (PRIVATE reads n weight sets, SHARED
            1 + (1 - s)(n - 1); the activation traffic cancels to first order,
            the premise V6 checks).
  W_g     = E x 2F x H x b (w1), E x H x F x b (w2), b the dtype's bytes.
  P       = npn = N / BLOCK_N (w1 N = 2F, w2 N = H): the N-tiles of one M-row.
  x       = (P - 1) BLOCK_N K b / L2_REF, K = H (w1) or F (w2), L2_REF = 60 MiB
            (rental 1's unit, so x is comparable across the registrations).
  W_c     = sm_count x the least of the four recorded occupancy limits, per
            GEMM, one value over the page's cells
            (`wave_split_bytes.page_window`).
  d       = P / W_c.
  far(n)  = the far share of SHARED's extra L2 hits by SUBTRACTION (rental 1):
            per cell, tex hits Hh, fabric sectors F, fabric misses
            Fm = M - (T - Hh) (M the read lookup misses, T the tex read
            sectors), far hits F - Fm; far = dF / (dHh + dF), d = SHARED - PRIVATE.
  far_direct(n)  the same share from the fabric's own read hits
            (`lts__t_sectors_srcunit_ltcfabric_op_read_lookup_hit.sum`, a
            `--partition-metrics` page only), None without them.
  partition(arm, n)  the cross-check: direct fabric read hits / (F - Fm).
  fabric_over_mn(n)  PRIVATE's F / Mn, Mn = T - Hh (the rental-1 control,
            [0.50, 0.56]).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

L2_REF_BYTES = 62914560
FABRIC = "lts__t_sectors_srcunit_ltcfabric.sum"
FABRIC_HIT = "lts__t_sectors_srcunit_ltcfabric_op_read_lookup_hit.sum"
TREADS = tuple(range(2, 10))


def gemm_shape(cfg, gemm: str) -> tuple[int, int]:
    """(K, N) of one GEMM."""
    if gemm == "w1":
        return cfg.hidden_size, 2 * cfg.intermediate_size
    return cfg.intermediate_size, cfg.hidden_size


def weight_bytes(cfg, gemm: str, b: int) -> int:
    k, n = gemm_shape(cfg, gemm)
    return cfg.num_experts * k * n * b


def _parts(pg: dict, rec: dict):
    """(tex hits, far hits by subtraction, direct far hits or None, Mn)."""
    t, hh, m = pg["l2_tex_read_sectors"], pg["l2_tex_read_hit_sectors"], pg["l2_read_miss_sectors"]
    f = rec[FABRIC]
    fm = m - (t - hh)
    return hh, f - fm, rec.get(FABRIC_HIT), t - hh, f


def survival(page: dict) -> dict:
    """{gemm: {x, d, P, W_c, ctas, s{n}, far{n}, far_direct{n}, partition{(arm,n)},
    fabric_over_mn{n}}} for one G=1 page, plus page-level `v6` (the page's V6
    verdict, None when it has none)."""
    design = page["design"]
    cfg = MODEL_CONFIGS[design["model"]]
    b = dtype_bytes(design["dtype"])
    bn = int(design["block_n"])
    cells = {(str(c["arm"]), int(c["n"])): c for c in page["cells"]}
    out: dict = {}
    for g in ("w1", "w2"):
        k, ncols = gemm_shape(cfg, g)
        P = ncols // bn
        wc, ctas = W.page_window(page, g)
        wg = weight_bytes(cfg, g, b)
        row = {"x": (P - 1) * bn * k * b / L2_REF_BYTES, "P": P, "W_c": wc, "ctas": ctas,
               "d": P / wc, "s": {}, "far": {}, "far_direct": {}, "partition": {},
               "fabric_over_mn": {}}
        for n in TREADS:
            if ("shared", n) not in cells or ("private", n) not in cells:
                continue
            sh, pr = cells[("shared", n)], cells[("private", n)]
            row["s"][n] = ((pr["per_gemm"][g]["dram_bytes_read"]
                            - sh["per_gemm"][g]["dram_bytes_read"]) / ((n - 1) * wg))
            hs, fs, ds, _mns, _fs = _parts(sh["per_gemm"][g], sh["recorded"][g])
            hp, fp, dp, mnp, ffp = _parts(pr["per_gemm"][g], pr["recorded"][g])
            dh, df = hs - hp, fs - fp
            row["far"][n] = df / (dh + df) if dh + df > 0 else None
            if ds is not None and dp is not None:
                dd = ds - dp
                row["far_direct"][n] = dd / (dh + dd) if dh + dd > 0 else None
                row["partition"][f"shared/{n}"] = ds / fs if fs else None
                row["partition"][f"private/{n}"] = dp / fp if fp else None
            else:
                row["far_direct"][n] = None
            row["fabric_over_mn"][n] = ffp / mnp if mnp else None
        out[g] = row
    v6 = [x.get("verdict") for x in page.get("gates") or []
          if (x.get("number") or x.get("tag") or x.get("id")) == "V6"]
    out["v6"] = v6[0] if v6 else None
    return out
