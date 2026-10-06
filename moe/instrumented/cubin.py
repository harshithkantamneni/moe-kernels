"""Static occupancy of a compiled fused_moe_kernel, read off its Triton cache files.

The perturbation gate's third leg (owner decision 2, 2026-10-06): the instrumented and
the plain kernel must have the SAME occupancy limit. Occupancy is a property of the
compiled config (registers per thread, shared memory per CTA, warps), not of the shape
(design-r4-review REVIEW.md section 0: the parsed register counts reproduce ncu's
launch__registers_per_thread on every published page). Pure Python: the ELF parse runs on
the laptop against the published cubins, and on the VM against the unit's own cache.

    regs = regcount(cubin_bytes)                     # EIATTR_REGCOUNT of .nv.info
    occ = occupancy(regs, shared_bytes, num_warps)   # CTAs per SM and the binding limit
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

#: sm_90 residency (CUDA occupancy calculator; PRINCIPLES.md section 1)
SM90 = {"regs_per_sm": 65536, "reg_unit_per_warp": 256, "max_regs_per_thread": 255,
        "smem_per_sm": 233472, "smem_reserved_per_cta": 1024, "max_warps": 64,
        "max_ctas": 32, "warp_alloc": 4}
EIATTR_REGCOUNT = 0x2F


def _sections(b: bytes) -> dict:
    if b[:4] != b"\x7fELF" or b[4] != 2:
        raise ValueError("not an ELF64 cubin")
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]

    def name(o):
        return b[stro + o: b.index(b"\0", stro + o)].decode()
    return {name(s[0]): (s[4], s[5]) for s in sh}


def regcount(b: bytes) -> int | None:
    """Registers per thread: EIATTR_REGCOUNT in a `.nv.info.<kernel>` section."""
    for name, (off, size) in _sections(b).items():
        if not name.startswith(".nv.info"):
            continue
        p = off
        while p < off + size:
            fmt, attr = b[p], b[p + 1]
            if fmt == 0x04:
                ln = struct.unpack_from("<H", b, p + 2)[0]
                if attr == EIATTR_REGCOUNT:
                    return struct.unpack_from("<I", b, p + 8)[0]
                p += 4 + ln
            else:
                p += 4
    return None


def occupancy(regs: int, shared: int, num_warps: int, card: dict = SM90) -> dict:
    """CTAs per SM under each limit and the least of them (registers, shared, warps, CTAs)."""
    per_warp = -(-regs * 32 // card["reg_unit_per_warp"]) * card["reg_unit_per_warp"]
    warps = -(-num_warps // card["warp_alloc"]) * card["warp_alloc"]
    lim = {"registers": card["regs_per_sm"] // (per_warp * warps) if regs else card["max_ctas"],
           "shared": card["smem_per_sm"] // (shared + card["smem_reserved_per_cta"]),
           "warps": card["max_warps"] // num_warps, "ctas": card["max_ctas"]}
    occ = min(lim.values())
    return {"ctas_per_sm": occ, "limits": lim,
            "binding": sorted(k for k, v in lim.items() if v == occ)}


def cache_entries(cache: Path | str) -> list[dict]:
    """Every compiled fused_moe_kernel under a Triton cache directory: its config (from
    the cache's json), registers (cubin), shared, the source file its TTIR names (plain:
    vLLM's fused_moe.py; instrumented: fused_moe_instr.py) and whether its PTX reads a
    timer (globaltimer / clock64)."""
    out = []
    for cub in sorted(Path(cache).rglob("fused_moe_kernel.cubin")):
        if not cub.with_suffix(".json").exists():
            continue
        j = json.loads(cub.with_suffix(".json").read_text())
        ptx = cub.with_suffix(".ptx")
        ptx_t = ptx.read_text() if ptx.exists() else ""
        src = ""
        for suf in (".source", ".ttir"):
            p = cub.with_suffix(suf)
            if p.exists():
                src = p.read_text()
                break
        origin = ("instrumented" if "fused_moe_instr.py" in src else
                  "plain" if "fused_moe.py" in src else "unknown")
        regs = regcount(cub.read_bytes())
        shared, warps = int(j.get("shared") or 0), int(j.get("num_warps") or 0)
        out.append({"dir": cub.parent.name, "origin": origin, "num_warps": warps,
                    "num_stages": j.get("num_stages"), "shared": shared, "regs": regs,
                    "timer_in_ptx": ("%globaltimer" in ptx_t) or ("%clock64" in ptx_t),
                    "eviction_in_ptx": ("evict_first" in ptx_t) or ("evict_last" in ptx_t),
                    "mma": ("wgmma" if "wgmma.mma_async" in ptx_t else
                            "mma.sync" if "mma.sync" in ptx_t else "?"),
                    "occupancy": occupancy(regs or 0, shared, warps) if warps else None})
    return out


# --------------------------------------------------------------------------
# a compiled kernel's facts, the L2 hint in PTX, and SASS (rental 4 review fixes)
# --------------------------------------------------------------------------

def compiled_facts(ck) -> dict | None:
    """Registers (from the cubin's EIATTR_REGCOUNT), shared, warps, PTX and cubin bytes of a
    Triton CompiledKernel (what `kernel[grid](...)` returns); None when there is none."""
    if ck is None:
        return None
    asm = getattr(ck, "asm", None) or {}
    md = getattr(ck, "metadata", None)

    def get(k):
        if md is None:
            return None
        return md.get(k) if isinstance(md, dict) else getattr(md, k, None)
    cub = asm.get("cubin")
    regs = regcount(cub) if isinstance(cub, (bytes, bytearray)) else getattr(ck, "n_regs", None)
    shared, warps = int(get("shared") or 0), int(get("num_warps") or 0)
    return {"regs": regs, "shared": shared, "num_warps": warps, "ptx": asm.get("ptx") or "",
            "cubin": cub, "occupancy": occupancy(regs or 0, shared, warps) if warps else None}


def hint_in_ptx(ptx: str) -> bool:
    """Whether a kernel's A/B copies carry an L2 eviction policy: some cp.async or global
    load line names `L2::cache_hint`. Triton 3.7.1 lowers a pipelined load to cp.async with
    .ca/.cg only and drops `eviction_policy` (build-r4-review section 1, E1), so a hinted
    unit is refused unless this reads True on its compiled PTX. `L1::evict_*` is not an L2
    policy and does not count."""
    return any("L2::cache_hint" in ln and ("cp.async" in ln or "ld.global" in ln)
               for ln in ptx.splitlines())


import re as _re  # noqa: E402

_SASS = _re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)\s*([^;]*);")


def parse_sass(text: str) -> list[dict]:
    """cuobjdump -sass lines as {addr, pred, op, args}, in address order."""
    out = []
    for m in _SASS.finditer(text):
        out.append({"addr": int(m.group(1), 16), "pred": m.group(2).strip(), "op": m.group(3),
                    "args": " ".join(m.group(4).split())})
    return out


def normalize_sass(text: str) -> list[str]:
    """The instruction stream with addresses, encodings and line info stripped: what must be
    equal between the plain kernel and the all-off copy."""
    return [f"{i['pred']} {i['op']} {i['args']}".strip() for i in parse_sass(text)]


def _is_clock(i: dict) -> bool:
    return i["op"].split(".")[0] in ("CS2R", "S2R", "S2UR") and ("SR_CLOCK" in i["args"] or "SR_GLOBALTIMER" in i["args"])


def sass_checks(text: str) -> dict:
    """Where the stamps' clock reads sit in a kernel's SASS (the scoring precondition):
    start_before_first_ldg: a clock read precedes the first LDG (num_tokens_post_padded);
    loop_clock_reads: clock reads inside the k-loop (the back-branch spanning the most HMMA);
    epi_bracket: a clock read between the loop-exit BAR and the first LDG after the loop
    (the topk_weights load), None when no LDG follows the loop (w1)."""
    ins = parse_sass(text)
    out = {"instructions": len(ins), "start_before_first_ldg": None, "loop_clock_reads": None,
           "epi_bracket": None, "loop": None}
    if not ins:
        return out
    ldg = [i["addr"] for i in ins if i["op"].startswith("LDG")]
    clk = [i["addr"] for i in ins if _is_clock(i)]
    out["start_before_first_ldg"] = bool(clk and ldg and min(clk) < min(ldg))
    loops = []
    for i in ins:
        m = _re.match(r"0x([0-9a-f]+)", i["args"]) if i["op"].startswith("BRA") else None
        if m and int(m.group(1), 16) < i["addr"]:
            lo, hi = int(m.group(1), 16), i["addr"]
            loops.append((sum(lo <= j["addr"] <= hi and j["op"].startswith("HMMA") for j in ins), lo, hi))
    if not loops:
        return out
    _n, lo, hi = max(loops)
    out["loop"] = [lo, hi]
    out["loop_clock_reads"] = sum(lo <= a <= hi for a in clk)
    bar = next((i["addr"] for i in ins if i["addr"] > hi and i["op"].startswith("BAR")), None)
    post_ldg = next((i["addr"] for i in ins if i["addr"] > hi and i["op"].startswith("LDG")), None)
    if post_ldg is not None:
        out["epi_bracket"] = bool(bar is not None and bar < post_ldg and any(bar < a < post_ldg for a in clk))
    return out
