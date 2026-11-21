#!/usr/bin/env python
"""Writes the decode stage's two registered JSON files and their text: the CAL rental's
(docs/registered/2026-10-14-decodeC-cal-gh200) and the blind rental's, frozen now and finalised
mechanically from the CAL score (docs/registered/2026-10-14-decodeB-blind-gh200).

    python scripts/scoring/decode/register_decode.py [--out-dir docs/registered] [--check] [--only cal|blind]
    python scripts/scoring/decode/register_decode.py --check-card     # recompute v4d's card constants (about 1 min)
    python scripts/scoring/decode/register_decode.py --mps-check      # the power simulation once more on torch MPS, float32

Every number is computed here from committed files (the decode pages of scripts/decode_synth.py, v4d
over the pinned v3 and v4 modules, the upstream record) or is a typed-in input with its provenance
and label. `--check` recomputes and compares (exit 1 on any difference); the `code_pins` block is
excluded from the comparison, as in rentals 3 to 7 and the tile stage. No decode page exists when
this runs; every published page is SEEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcommon as DC  # noqa: E402
import regdecode as RG  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
#: every pin of the tile stage (byte-identical, tests/test_pinned_files.py and test_stage_decode_pins.py) and the decode stage's own code
TILE_PINS = tuple(json.loads((REG / "2026-10-14-tileA-e1cal-gh200.json").read_text())["code_pins"])
DECODE_CODE = ("scripts/r3_decode.py", "scripts/decode_synth.py", "scripts/r3_timing_model_v4d.py",
               "scripts/dram_counter_route_decode.py", "scripts/gh200_decode_session.sh",
               "scripts/scoring/decode/dcommon.py", "scripts/scoring/decode/regdecode.py",
               "scripts/scoring/decode/finalise_decode.py", "moe/instrumented/UPSTREAM_DECODE.json")
PINNED = tuple(dict.fromkeys(TILE_PINS + DECODE_CODE))
SKIP_TXT = {"code_pins", "nominal_cells", "files", "cells", "scorer_ambiguities"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT:
                out.append(f"{prefix}{k}: (in the JSON)")
                continue
            if isinstance(x, (dict, list)) and depth < 2:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {x if isinstance(x, str) else json.dumps(x, default=str)}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {x if isinstance(x, str) else json.dumps(x, default=str)}")
    else:
        out.append(f"{prefix}{v}")


def txt(part: str, d: dict) -> list[str]:
    which = "the CAL rental" if part == "cal" else "the blind rental (frozen; finalised mechanically)"
    L = [f"REGISTERED {d['registered']}: the decode stage, {which}, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    L += RG.TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


def build(only=None) -> dict:
    parts = [p for p in DC.PARTS if only is None or p in only]
    docs = {p: RG.BUILDERS[p]() for p in parts}
    pins = {p: sha256(REPO / p) for p in PINNED}
    for d in docs.values():
        d["code_pins"] = pins
    return docs


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def check_card() -> int:
    sys.path[:0] = [str(REPO / "scripts" / "scoring" / "rental6")]
    import reg6 as R6
    x = list(R6.pricer().f2.x)
    got = {"T0_eager_ms": x[0], "c_ns": x[1], "bw_gbps": x[2], "s_small_eager_ms": x[3], "s_block_eager_ms": x[4]}
    bad = {k: (v, DC.V4D.CARD[k]) for k, v in got.items() if float(v) != DC.V4D.CARD[k]}
    print("card constants:", "same" if not bad else f"DIFFER {bad}")
    return 1 if bad else 0


def mps_check() -> int:
    """The CAL power once more on the laptop GPU (torch MPS, float32) against the CPU float64 result."""
    import math

    import torch
    if not torch.backends.mps.is_available():
        print("no MPS device: nothing to check")
        return 0
    d = json.loads((REG / f"{DC.NAMES['cal']}.json").read_text())
    cells = [c for c in d["nominal_cells"] if c.get("ms_nominal_fixed")]
    pred = torch.tensor([c["ms_nominal_fixed"] for c in cells], dtype=torch.float32, device="mps")
    sig = torch.tensor([RG.POWER_SIGMA_SMALL if c["ms_nominal_fixed"] < 0.05 else RG.POWER_SIGMA for c in cells],
                       dtype=torch.float32, device="mps")
    g = torch.Generator(device="mps").manual_seed(RG.POWER_SEED)
    out = {}
    bars = DC.BARS_PROVISIONAL
    for name, fn in RG.SCENARIOS.items():
        bias = torch.tensor([fn(c) for c in cells], dtype=torch.float32, device="mps")
        truth = pred * (1 + bias)
        holds = 0
        for _ in range(RG.POWER_DRAWS // 100):
            meas = truth * (1 + torch.randn((100, pred.numel()), generator=g, device="mps") * sig)
            dd = pred - meas
            e = torch.where(dd.abs() * 1e3 <= DC.FLOOR_US, torch.zeros_like(dd), dd / meas)
            ok = ((e * e).mean(1).sqrt() <= bars["rms_max"]) & (e.abs().amax(1) <= bars["max_abs"]) & (e.mean(1).abs() <= bars["mean_max"])
            holds += int(ok.sum())
        out[name] = holds / RG.POWER_DRAWS
    cpu = d["power"]["p_holds"]
    worst = max(abs(out[k] - cpu[k]) for k in out)
    se = max(math.sqrt(max(p * (1 - p), 1e-9) / RG.POWER_DRAWS) for p in cpu.values())
    print(f"MPS float32 {out}\nCPU float64 {cpu}\nlargest difference {worst:.4f} (Monte Carlo se about {se:.4f})")
    return 0 if worst <= 4 * se + 1e-3 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REG)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="+", choices=DC.PARTS, default=None)
    ap.add_argument("--check-card", action="store_true")
    ap.add_argument("--mps-check", action="store_true")
    a = ap.parse_args(argv)
    if a.check_card:
        return check_card()
    if a.mps_check:
        return mps_check()
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{DC.NAMES[part]}.json"
        tpath = a.out_dir / f"{DC.NAMES[part]}.txt"
        jtext = json.dumps(doc, indent=1, default=str) + "\n"
        ttext = "\n".join(txt(part, doc)) + "\n"
        if a.check:
            same_j = jpath.exists() and _strip(json.loads(jpath.read_text())) == _strip(json.loads(jtext))
            same_t = tpath.exists() and tpath.read_text() == ttext
            for p, same in ((jpath, same_j), (tpath, same_t)):
                print(f"{p.name}: {'same' if same else 'DIFFERS'}")
                bad += not same
        else:
            jpath.write_text(jtext)
            tpath.write_text(ttext)
            print(f"wrote {jpath}\nwrote {tpath}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
