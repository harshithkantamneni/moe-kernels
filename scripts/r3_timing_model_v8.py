#!/usr/bin/env python
"""MODEL v8: v7 made to price real vLLM 0.27.1 fused-MoE calls (design 2026-11-19, moe-kernels-work/2026-11-19/v8:
DESIGN.md, INVENTORY.md; critique disposition items B3, M2, M3, M4, M5 and MINOR).

ONE ENTRY POINT, `price(call)`. A call is what one fused_experts launch on GH200 sees:

    {model: a MODEL_CONFIGS key or a moe.spec.MoEConfig (registered in-process by `register_model`),
     T: the padded token count, counts: per-expert routed rows (len E, sum == T k),
     ids: token-major topk ids (len T k) or None,
     tile / G: optional; default vLLM 0.27.1's default ladder, moe.bench.tile_resolve.default_config(T, E),
     sag128, sag64, knee: REQUIRED on a prefill tile (no default; a missing one is a ValueError): the board's 128x128
           sag par at 1410 MHz, the 64x128 par and the K1 knee, each a pair (review 1, BLOCKER 2)}

priced on the six default tiles: the four decode tiles by v7's decode branch (v6d + D6FIX at the v7 decode blind's
frozen constants: finalise_decode.params_of on the frozen block of
docs/registered/2026-11-17-v7-decode-blind-gh200.json,
its terms and law, prm_fix by v7bcommon.prm_fix_of) and the two prefill tiles by v7's prefill branch (round 5 + RR).
INVARIANT: on every input v7 accepts (whole-tread page cells, no ids), v8's number is v7's bit for bit. v8 changes no
fitted constant, no law and no term.

THE SWITCHES (each its own, owner rule 2026-10-14), beside v7's RR and D6FIX:
  GEOM  every use of the tread n on the prefill paths re-expressed through numel = sum(counts) and counts: grid rows
        ceil((numel + D (BM - 1)) / BM), A bytes numel x row bytes, align_path(numel, D), the walk's guard numel >= 1,
        the histogram check sum(counts) == numel; n kept as a label only. HOPPER_G1_N read at n_eff = numel / (E BM)
        >= 6 (refused, as v7). v7's latent defect (64x128 G 1 n >= 6: KeyError('reads') in price_parts) is a refusal
        (ms None, why). Off: v7's own price at an integer tread, any other call refused.
  IDS   real topk ids replace realize_counts + the seeded shuffle on prefill (wave_split_bytes_v3.
        tile_content_ids_counts) and V4DP.ids_of on the decode naive path; a call without ids keeps the generated ids.
        Off: ids ignored.
  CACHE (always on, not a model feature) every cache key that held n holds the Tread, whose hash is a digest of
        (numel, counts, ids, order); k5._READS keys are v7's (model, page, n, label, BM, G) with n its label, extended
        by that digest.

HOW (frozen files untouched): `installed()` swaps the inventoried leaf functions (SWAPS below) for counts-based copies
at EVERY binding (critique M2; the grep-complete list is in SWAP_BINDINGS) and restores them on exit (single-threaded
by design). Each copy dispatches to the frozen original unless it is handed a `Tread`, so v7's r3 and decode paths and
any v7 caller run the originals. On the prefill path n reaches the frozen code as a `Tread`: a POISONED sentinel that
raises PoisonedTread on int(), index, float, bool, arithmetic and any comparison with a non-Tread, so a use v8 did not
replace fails loudly instead of truncating. Each copy names the frozen file:line and the exact expression changed.

KNOWN ASSUMPTIONS (DESIGN.md, carried): within-expert token order ascending (the walk's; vLLM's block-scan align is
atomic-arrival order; `order_seed` on a call shuffles it, for the offline order-sensitivity check only); numel < E
(T k < E) is decode territory; fitted constants at new geometry are extrapolation, not refit; B_OTHER's
(E BM // k) floor is inexact at k 6 (about 0.02%), unchanged from v7.

ERNIE-4.5-21B-A3B (`ERNIE`) is supplied here at runtime and registered into MODEL_CONFIGS in this process only;
moe/spec.py is not edited (it is sha-pinned by registrations). Its numbers were checked by hand against the HF
config.json on 2026-11-19 (moe/spec.py's docstring names scripts/verify_model_configs.py, which does not exist).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_timing_model_v7 as V7  # noqa: E402
import round5_chain as R5  # noqa: E402

from moe.spec import MODEL_CONFIGS, MoEConfig  # noqa: E402

Refused = V7.Refused
DECODE_REG = REPO / "docs" / "registered" / "2026-11-17-v7-decode-blind-gh200.json"
PREFILL_REG = REPO / "docs" / "registered" / "2026-11-18-v7-prefill-blind-gh200.json"
PREFILL_SCORE = REPO / "scripts" / "scoring" / "v7pblind" / "prefill_blind.score.json"

# ================================================================== the new model, supplied at runtime

ERNIE = MoEConfig(name="ernie-4.5-21b-a3b", hidden_size=2560, intermediate_size=1536, num_experts=64, top_k=6,
                  gate_fn="softmax", norm_topk_prob=True, shared_experts=2, num_layers=28, first_moe_layer=1,
                  hf_repo="baidu/ERNIE-4.5-21B-A3B-PT", verified=True)


def register_model(cfg: MoEConfig) -> str:
    """Put cfg into MODEL_CONFIGS in this process only (moe/spec.py untouched); refuses a different config under a
    name already present."""
    have = MODEL_CONFIGS.get(cfg.name)
    if have is not None and have != cfg:
        raise Refused(f"{cfg.name}: MODEL_CONFIGS already holds a different config")
    MODEL_CONFIGS[cfg.name] = cfg
    return cfg.name


# ================================================================== switches

@dataclass(frozen=True)
class Switches:
    RR: bool = True       # v7: prefill w1 re-read term at G 1
    D6FIX: bool = True    # v7: decode D6-24 fix
    GEOM: bool = True     # v8: n -> numel and counts on the prefill paths
    IDS: bool = True      # v8: real topk ids replace the generated ids

    def v7(self) -> V7.Switches:
        return V7.Switches(RR=self.RR, D6FIX=self.D6FIX)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


V8 = Switches()
#: v7 exactly on a page cell (GEOM off: v7's own price at an integer tread; IDS off)
V7_ONLY = Switches(GEOM=False, IDS=False)


# ================================================================== the poisoned tread and the real ids

class PoisonedTread(BaseException):
    """A frozen site used the tread n in arithmetic, a conversion or a comparison v8 has not replaced. A
    BaseException, so no `except Exception` in the frozen code can swallow it."""


def _poison(name):
    def f(self, *a, **k):
        raise PoisonedTread(f"{name} on {self!r}: a use of the tread n that v8 has not replaced")
    f.__name__ = name
    return f


class _Poisoned:
    """Hashable, printable, and nothing else: every numeric conversion, arithmetic and ordering raises."""
    __slots__ = ()
    _key: tuple

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        if type(other) is type(self):
            return self._key == other._key
        raise PoisonedTread(f"== between {self!r} and {other!r}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __array__(self, *a, **k):
        raise PoisonedTread(f"numpy conversion of {self!r}")


for _name in ("__int__", "__index__", "__float__", "__complex__", "__bool__", "__round__", "__trunc__", "__floor__",
              "__ceil__", "__abs__", "__neg__", "__pos__", "__invert__", "__lt__", "__le__", "__gt__", "__ge__",
              "__len__", "__iter__", "__getitem__", "__contains__",
              *(f"__{p}{op}__" for op in ("add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod",
                                            "pow", "lshift", "rshift", "and", "or", "xor") for p in ("", "r", "i"))):
    setattr(_Poisoned, _name, _poison(_name))


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode() + str(p.shape).encode() + np.ascontiguousarray(p).tobytes())
        else:
            h.update(json.dumps(p, separators=(",", ":"), default=str).encode())
        h.update(b"|")
    return h.hexdigest()


class RealIds(_Poisoned):
    """A call's token-major topk ids (T x k), carried where the frozen code carries a page's shuffle seed (so it
    reaches wave_split_bytes_v3.tile_content_ids_counts unchanged in between). order_seed None: each expert's
    tokens ascending (the walk's assumption); an int: shuffled per expert by numpy default_rng(order_seed) (the
    offline order-sensitivity check)."""
    __slots__ = ("ids", "T", "k", "order_seed", "digest", "_key")

    def __init__(self, ids, T: int, k: int, order_seed: int | None = None):
        a = np.asarray(ids, dtype=np.int64).reshape(-1)
        if a.size != int(T) * int(k):
            raise ValueError(f"ids: {a.size} entries, not T k = {int(T) * int(k)}")
        a.setflags(write=False)
        self.ids, self.T, self.k, self.order_seed = a, int(T), int(k), order_seed
        self.digest = _digest("ids", a, order_seed)
        self._key = ("RealIds", self.digest)

    def __repr__(self):
        return f"RealIds(T={self.T}, k={self.k}, order={self.order_seed}, {self.digest[:12]})"


class Tread(_Poisoned):
    """The tread n as v8 hands it to the frozen code: a label (`label`, the page's n on a page cell, else n_eff) that
    carries numel = sum(counts), the counts and a digest of (counts, ids); every use as a number raises. str() is the
    label (v7's Cell.key 'arm/G{G}/n{n}' stays v7's string on a page cell)."""
    __slots__ = ("numel", "counts", "E", "bm", "label", "digest", "_key")

    def __init__(self, counts, E: int, bm: int, *, label=None, seed=None):
        c = tuple(int(v) for v in counts)
        if len(c) != int(E):
            raise ValueError(f"{len(c)} counts on E {E}")
        self.counts, self.E, self.bm = c, int(E), int(bm)
        self.numel = int(sum(c))
        if self.numel < 1:
            raise Refused("numel 0: no routed row (v8's guard numel >= 1)")
        whole = self.numel % (self.E * self.bm) == 0
        self.label = label if label is not None else (self.numel // (self.E * self.bm) if whole else None)
        self.digest = _digest("tread", c, self.bm, seed.digest if isinstance(seed, RealIds) else seed)
        self._key = ("Tread", self.numel, c, self.E, self.bm, self.digest)

    @property
    def n_eff(self) -> float:
        return self.numel / (self.E * self.bm)

    def whole(self) -> bool:
        return self.numel % (self.E * self.bm) == 0

    def hopper_g1(self, experts: int, block_m: int) -> bool:
        """HOPPER_G1_N at n_eff: numel >= HOPPER_G1_N E BM, in integers (equal to n >= 6 at whole n)."""
        assert (experts, block_m) == (self.E, self.bm), ((experts, block_m), (self.E, self.bm))
        return self.numel >= _W().HOPPER_G1_N * experts * block_m

    def __str__(self):
        return str(self.label) if self.label is not None else f"{self.n_eff:.6g}"

    def __format__(self, spec):
        return format(str(self), spec)

    def __repr__(self):
        return f"Tread(n={self}, numel={self.numel}, E={self.E}, BM={self.bm})"


class Tread32(_Poisoned):
    """WV4.price's A-proxy tread n x BM / 32 (wave_split_bytes_v4.py:171), as a poisoned label: NATIVE returns A 0
    before reading it (scoring/rental6/r6model.py:88-89); any other use raises."""
    __slots__ = ("of", "_key")

    def __init__(self, of: Tread):
        self.of, self._key = of, ("Tread32", of._key)

    def __str__(self):
        return f"{self.of.numel / (self.of.E * 32):.6g}"

    def __repr__(self):
        return f"Tread32({self})"


# ================================================================== the swaps

def _W():
    import wave_split_bytes as W
    return W


class _NS:
    """Attribute access to a frozen function's module globals, read at call time (a popped work module)."""

    def __init__(self, fn):
        object.__setattr__(self, "_g", fn.__globals__)

    def __getattr__(self, k):
        return self._g[k]


_ORIG: dict = {}
STATS: Counter = Counter()      # Tread-path calls of each copy (the canary reads it)


def _orig(name):
    return _ORIG[name]


# ---- wave_split_bytes.py:434 Geometry.num_pid_m
def num_pid_m(self, arm, n):
    if not isinstance(n, Tread):
        return _orig("Geometry.num_pid_m")(self, arm, n)
    STATS["Geometry.num_pid_m"] += 1
    d = dict(self.declared)[arm]
    # wave_split_bytes.py:438  rows = self.experts * n * self.block_m + d * (self.block_m - 1)
    #                     ->  rows = numel + d * (self.block_m - 1)                  (vLLM's EM = numel + D (BM - 1))
    assert (self.experts, self.block_m) == (n.E, n.bm), (self.experts, self.block_m, n)
    rows = n.numel + d * (self.block_m - 1)
    return -(-rows // self.block_m)


# ---- wave_split_bytes.py:632 walk (the frozen original; W.walk itself is rebound to r4.walk_law by r4.py:184)
def walk(arm, G, n, P, W_c, num_pid_m, experts, *, slab_bytes=0, atile_bytes=0, counts=None, block_m=0):
    if not isinstance(n, Tread):
        return _orig("W.walk")(arm, G, n, P, W_c, num_pid_m, experts, slab_bytes=slab_bytes, atile_bytes=atile_bytes,
                               counts=counts, block_m=block_m)
    STATS["W.walk"] += 1
    W = _W()
    # wave_split_bytes.py:644  if min(G, n, P, num_pid_m, experts) < 1
    #                     ->  if min(G, P, num_pid_m, experts) < 1 or numel < 1
    if min(G, P, num_pid_m, experts) < 1 or n.numel < 1:
        raise ValueError(f"walk needs G, P, num_pid_m and E >= 1 and numel >= 1, got {G}, {P}, {num_pid_m}, "
                         f"{experts}, numel {n.numel}")
    # wave_split_bytes.py:648-662  the counts branch only (`live_m = experts * n` on a balanced cell -> refused: v8
    # always walks a histogram)
    if counts is None:
        raise PoisonedTread("walk: a balanced walk (live_m = experts * n) with a Tread; v8 walks histograms only")
    if arm == "private":
        raise W.Refused("PRIVATE has no per-expert walk: its copies are one per M-tile of a balanced expert")
    if len(counts) != experts or block_m < 1:
        raise ValueError(f"a histogram of {len(counts)} experts on E={experts}, BLOCK_M {block_m}")
    t_owner, t_rows = W.tile_owners(counts, block_m)
    live_m = int(t_owner.size)
    if num_pid_m < live_m:
        raise ValueError(f"num_pid_m {num_pid_m} is below the {live_m} live M-tiles")
    pid = np.arange(num_pid_m * P, dtype=np.int64)
    gid, pid_m, pid_n = W.pid_map(pid, num_pid_m, P, G)
    keep = pid_m < live_m
    gid, pid_m, pid_n = gid[keep], pid_m[keep], pid_n[keep]
    live = int(pid_m.size)
    owner = t_owner[pid_m]
    key = owner * P + pid_n
    cur, prev = W._previous(key)
    slabs = live - int(cur.size)
    D = cur - prev
    inside = gid[cur] == gid[prev]
    win1 = cur < W_c
    xc = cur[~inside]
    # wave_split_bytes.py:682  stack_distance(G, n, ...) with tile_owner given: n is not read (its `// n` is the
    # tile_owner None branch, :612); passed through as the frozen walk passes it
    S = (W.stack_distance(G, n, P, live_m, slab_bytes, atile_bytes, gid[xc], pid_n[xc], t_owner)
         if xc.size else np.zeros(0))
    x = Counter(zip(D[~inside].tolist(), win1[~inside].tolist(), S.tolist(), strict=True))
    in_win1 = int(np.sum(inside & win1))
    in_later = int(np.sum(inside & ~win1))
    groups = -(-num_pid_m // G)
    ws = np.zeros(groups)
    for g in range(groups):
        first = g * G
        tiles = range(first, min(first + G, live_m)) if first < live_m else range(0)
        owners = len({int(t_owner[m]) for m in tiles})
        ws[g] = len(tiles) * atile_bytes + owners * slab_bytes
    cur_a, prev_a = W._previous(pid_m)
    da = cur_a - prev_a
    wa = cur_a < W_c
    a = Counter()
    frac = (t_rows[pid_m[cur_a]] / block_m).tolist()
    for key_a, f in zip(zip(da.tolist(), wa.tolist(), ws[gid[cur_a]].tolist(), strict=True), frac, strict=True):
        a[key_a] += f

    def arrays(counter, width):
        if not counter:
            return [np.zeros(0) for _ in range(width + 1)]
        keys = sorted(counter)
        cols = [np.array([k[i] for k in keys], dtype=float) for i in range(width)]
        return cols + [np.array([counter[k] for k in keys], dtype=float)]

    x_D, x_w, x_S, x_k = arrays(x, 3)
    a_D, a_w, a_ws, a_k = arrays(a, 3)
    return W.Events(arm=arm, G=G, n=n, gemm="", P=P, W_c=W_c, num_pid_m=num_pid_m,
                    live=live, slabs=slabs, x_D=x_D, x_win1=x_w.astype(bool), x_k=x_k, x_S=x_S,
                    in_win1=in_win1, in_later=in_later, a_D=a_D,
                    a_win1=a_w.astype(bool), a_ws=a_ws, a_k=a_k)


# ---- wave_split_bytes.py:775 cell_events (the frozen original; W.cell_events is rebound by r4.py:185, r5.py:155,
# r7.py:70)
def cell_events(geom, arm, G, n, gemm, counts=None):
    if not isinstance(n, Tread):
        return _orig("W.cell_events")(geom, arm, G, n, gemm, counts)
    STATS["W.cell_events"] += 1
    W = _W()
    P = geom.get("P", gemm)
    Wc = geom.get("W_c", gemm)
    npm = geom.num_pid_m(arm, n)
    if counts is None:
        raise PoisonedTread("cell_events: a balanced cell with a Tread (closed form and DCR.r3_exposure read n)")
    # wave_split_bytes.py:786  if int(sum(counts)) != geom.experts * n * geom.block_m
    #                     ->  if int(sum(counts)) != numel                   (kept as an integrity guard)
    if int(sum(counts)) != n.numel:
        raise W.Refused(f"a histogram of {int(sum(counts))} rows: the call holds numel {n.numel}")
    # wave_split_bytes.py:789 `walk(...)`: the module's walk, read at call time (r4.walk_law), as the original reads it
    ev = W.walk(arm, G, n, P, Wc, npm, geom.experts, slab_bytes=geom.slab(gemm),
                atile_bytes=geom.atile(gemm), counts=counts, block_m=geom.block_m)
    if W.LATER_MISS and G >= 2 and ev.x_D.size:
        later = ~ev.x_win1 & (ev.x_S > geom.l2_bytes)
        ev = dataclasses.replace(ev, slabs=ev.slabs + int(ev.x_k[later].sum()),
                                 x_D=ev.x_D[~later], x_win1=ev.x_win1[~later],
                                 x_k=ev.x_k[~later], x_S=ev.x_S[~later])
    return dataclasses.replace(ev, gemm=gemm, per_set=geom.per_set(gemm),
                               weight_bytes=geom.get("W", gemm), atile=geom.atile(gemm),
                               ksteps=geom.get("K", gemm) // max(1, geom.block_k))


# ---- wave_split_bytes.py:1231 Model.domain
def domain(self, arm, G, n, gemm):
    if not isinstance(n, Tread):
        return _orig("Model.domain")(self, arm, G, n, gemm)
    STATS["Model.domain"] += 1
    W = _W()
    if gemm == "w2" and arm != "private" and G >= W.OUT_OF_DOMAIN_G:
        return W.OUT_OF_DOMAIN
    # wave_split_bytes.py:1236  and n >= HOPPER_G1_N
    #                      ->  and n_eff >= HOPPER_G1_N, i.e. numel >= HOPPER_G1_N E BM (Tread.hopper_g1)
    if (self.geom.hopper and gemm == "w2" and arm != "private" and G == 1
            and n.hopper_g1(self.geom.experts, self.geom.block_m)):
        return W.ILL_POSED
    return None


# ---- wave_split_bytes.py:1241 Model.evaluate
def evaluate(self, prm, cells):
    cells = list(cells)
    if not any(isinstance(c[2], Tread) for c in cells):
        return _orig("Model.evaluate")(self, prm, cells)
    STATS["Model.evaluate"] += 1
    W = _W()
    extra = []
    for c in cells:
        arm, G, n, gemm = c[:4]
        # wave_split_bytes.py:1250  range(1, n) lower treads: only for a balanced (len 4) cell; a Tread never is one
        if G == 1 and len(c) == 4:
            raise PoisonedTread(f"evaluate: a balanced G-1 cell {c!r} with a Tread (the monotonicity guard reads n)")
    allc = list(dict.fromkeys(cells + extra))
    b = self.batch(allc)
    q = b.solve(prm)
    qh = b.solve(prm, high=True)
    kap = b.kappa(prm, q)
    at = {c: i for i, c in enumerate(allc)}
    out = []
    for c in cells:
        i = at[c]
        arm, G, n, gemm = c[:4]
        why = []
        if kap[i] > W.KAPPA_MAX:
            why.append(f"kappa {kap[i]:.1f} > {W.KAPPA_MAX:g}")
        if abs(qh[i] - q[i]) > W.ROOT_GAP_MAX:
            why.append(f"roots {q[i]:.3f} and {qh[i]:.3f}")
        # wave_split_bytes.py:1264  if G == 1 and n > 1 and len(c) == 4  ->  never true for a len-6 Tread cell
        rule = self.domain(arm, G, n, gemm)
        out.append({"cell": c, "q": float(q[i]), "q_high": float(qh[i]),
                    "kappa": float(kap[i]), "guard": why, "rule": rule,
                    "ill": bool(why) or rule == W.ILL_POSED})
    return out


# ---- wave_split_bytes_v3.py:22 tile_content_ids_counts (the IDS switch)
_CONTENT_REAL: dict = {}


def tile_content_ids_counts(model, counts, block_m, shuffle_seed):
    if not isinstance(shuffle_seed, RealIds):
        return _orig("WV3.tile_content_ids_counts")(model, counts, block_m, shuffle_seed)
    STATS["WV3.tile_content_ids_counts"] += 1
    W = _W()
    key = (model, tuple(int(v) for v in counts), int(block_m), shuffle_seed)
    if key not in _CONTENT_REAL:
        cfg = MODEL_CONFIGS[model]
        # wave_split_bytes_v3.py:36-41  tokens = sum // k; ids = realize_counts(counts, tokens, k)[randperm(seed)]
        #                          ->  the call's own token-major topk ids, T x k
        tokens = int(sum(key[1])) // cfg.top_k
        if (shuffle_seed.T, shuffle_seed.k) != (tokens, cfg.top_k):
            raise W.Refused(f"{model}: ids of T {shuffle_seed.T} k {shuffle_seed.k} on counts of {tokens} tokens")
        ids = shuffle_seed.ids.reshape(tokens, cfg.top_k)
        rng = None if shuffle_seed.order_seed is None else np.random.default_rng(int(shuffle_seed.order_seed))
        seen: dict = {}
        cid, rows = [], []
        for e in range(cfg.num_experts):
            toks = np.nonzero((ids == e).any(axis=1))[0]
            if toks.size != key[1][e]:
                raise W.Refused(f"{model}: expert {e} holds {toks.size} rows in the ids, not {key[1][e]}")
            if rng is not None:
                toks = rng.permutation(toks)
            for j in range(-(-toks.size // block_m)):
                ch = tuple(toks[j * block_m:(j + 1) * block_m].tolist())
                cid.append(seen.setdefault(ch, len(seen)))
                rows.append(len(ch))
        _CONTENT_REAL[key] = (np.array(cid, dtype=np.int64), np.array(rows, dtype=np.int64))
    return _CONTENT_REAL[key]


# ---- wave_split_bytes_v4.py:116 reads_of
def reads_of(q, n):
    if not isinstance(n, Tread):
        return _orig("WV4.reads_of")(q, n)
    STATS["WV4.reads_of"] += 1
    import r3_timing_model as TM
    out = {}
    for g in TM.GEMMS:
        op = TM.BYTE_MODEL[f"operand_per_tile_{g}"]
        per = TM.IDS_PER_TREAD
        assert (TM.E, TM.BLOCK_M) == (n.E, n.bm) and op % per == 0, (TM.E, TM.BLOCK_M, n, op, per)
        # wave_split_bytes_v4.py:118  n * TM.BYTE_MODEL[f"operand_per_tile_{g}"]
        #                        ->  numel * (operand_per_tile // (E BLOCK_M))   (row bytes; an integer identity)
        out[g] = q[g] * TM.BYTE_MODEL[f"W_{g}"] + n.numel * (op // per)
    return out


# ---- wave_split_bytes_v4.py:154 price
def wv4_price(pr, model, arm, n, counts, copies, seed, t, G, prm, *, key="CLAMP", A_us=None, reads=None,
              rowkey_w1=None, proxy_n=None):
    if not isinstance(n, Tread):
        return _orig("WV4.price")(pr, model, arm, n, counts, copies, seed, t, G, prm, key=key, A_us=A_us, reads=reads,
                                  rowkey_w1=rowkey_w1, proxy_n=proxy_n)
    STATS["WV4.price"] += 1
    if arm != "native":
        # wave_split_bytes_v4.py:171-172 hands n32 to r6model.proxy_A_us, which formats str(n) into its table key on a
        # non-native arm (scoring/rental6/r6model.py:88-95): a Tread's label there would price another tread's A.
        # v8 prices real calls, which are NATIVE: any other arm on a Tread is refused (review 1, MINOR)
        raise Refused(f"v8 prices the native arm only on a call (arm {arm!r}: r6model's A proxy keys by str(n))")
    WV4 = _NS(_orig("WV4.price"))
    TM, V4 = WV4.TM, WV4.V4
    import r3_timing_model_v3 as TV3
    r6dir = str(REPO / "scripts" / "scoring" / "rental6")
    if r6dir not in sys.path:
        sys.path.insert(0, r6dir)
    import r6model as M6
    occ = {g: prm.occ(t, g) for g in TM.GEMMS}
    phis = None if t == V4.R3_TILE else {g: V4.phi(t, g, prm) for g in TM.GEMMS}
    scale = {g: V4.c_scale(t, g, prm) for g in TM.GEMMS}
    how = "page"
    if A_us is None:
        # wave_split_bytes_v4.py:171  n32 = proxy_n if proxy_n is not None else n * t.bm // 32
        #                        ->  n32 = proxy_n if proxy_n is not None else Tread32(n)   (a poisoned label)
        n32 = proxy_n if proxy_n is not None else Tread32(n)
        A_us, how = WV4.proxy_A_us(M6, pr.proxy, model, arm, n32, copies)
        if A_us is None and arm != "native":
            A_us, how = 0.0, "no proxy table: 0, replaced by the page's own A"
    ct = None if counts is None else tuple(int(v) for v in counts)
    with V4.at_tile(model, t, phis):
        q = None
        if reads is None:
            q = WV4.q_v3(pr.eng, model, n, ct, copies, seed, t, G, occ)
            if q is None:
                return {"ms": None, "ill": True, "key": key}
            reads = WV4.keyed_reads(key, q, n, WV4.q_ideal(model, n, ct, t.bm), rowkey_w1)
            if reads is None:
                return {"ms": None, "ill": False, "key": key, "why": "no priced row-key w1 row"}
        ctx = pr.ctx if occ == dict(pr.ctx.occupancy) else dataclasses.replace(pr.ctx, occupancy=dict(occ))
        cell = V4.make_cell(pr.eng, model, arm, n, ct, copies, G, t, reads, occ, pr.f2.k_w, ctx.sms)
        npath = V4.native_path_of(model, n, t)
        A = None if A_us is None else A_us * 1e-3
        with TM.dead_model("v2"):
            ms = V4.call_ms(pr.f2.x, cell, ctx, pr.f2.k_w, scale, native_path=npath, align_ms=A)
            parts = {}
            for g in TM.GEMMS:
                cg = float(pr.f2.x[1]) * scale[g]
                parts[g] = {"dead_ms": TM.dead_ms(cell.declared, n, g, cg, pr.f2.k_w, occ[g], ct),
                            "dead_ctas": TM.dead_ctas(cell.declared, n, g, ct),
                            "live_ctas": TM.window(arm, cell.declared, G, n, g,
                                                   TM.window_width(pr.f2.k_w, ctx.sms, occ[g]), ct).live,
                            "ksteps": TM.GEOMETRY[g].ksteps, "phi": TM.CTA_FIXED_KSTEPS[g]}
        b = TV3.v3_b_ms(arm, npath, A, key=cell.key)
    return {"ms": ms, "ill": False, "key": key, "reads": reads, "q": q, "A_us": A_us, "A_how": how,
            "path": cell.path, "native_path": npath, "declared": cell.declared, "occ": occ, "c_scale": scale,
            "phi": phis or dict(TM.CTA_FIXED_KSTEPS), "term_b_ms": b, "gemm": parts}


# ---- r3_timing_model_v4.py:397 make_cell
def make_cell(eng, model, arm, n, counts, copies, G, t, reads, occ, k_w, sms):
    if not isinstance(n, Tread):
        return _orig("V4.make_cell")(eng, model, arm, n, counts, copies, G, t, reads, occ, k_w, sms)
    STATS["V4.make_cell"] += 1
    if arm != "native":
        raise Refused(f"v8 prices the native arm only on a call (make_cell, arm {arm!r})")
    import private_weight_reference as PWR
    import r3_timing_model as TM
    cfg = MODEL_CONFIGS[model]
    D = cfg.num_experts if arm == "native" else cfg.num_experts * copies
    ct = None if counts is None else tuple(int(v) for v in counts)
    # r3_timing_model_v4.py:407  numel = PWR.ids_for_tread(cfg, n, t.bm)  ->  numel = sum(counts)
    numel = n.numel
    path = PWR.align_path(numel, D)
    sigma = {}
    for g in TM.GEMMS:
        w = TM.window_width(k_w, sms, occ[g])
        sigma[g] = reads[g] / TM.window(arm, D, G, n, g, w, ct).reads
    return TM.Cell(arm=arm, G=G, n=n, ms=float("nan"), mhz=None, path=path, declared=D, runs=(),
                   source="PREDICTED", reads=reads, sigma=sigma, fit=False, counts=ct)


# ---- r3_timing_model_v4.py:417 native_path_of
def native_path_of(model, n, t):
    if not isinstance(n, Tread):
        return _orig("V4.native_path_of")(model, n, t)
    STATS["V4.native_path_of"] += 1
    import private_weight_reference as PWR
    cfg = MODEL_CONFIGS[model]
    # r3_timing_model_v4.py:422  PWR.align_path(PWR.ids_for_tread(cfg, n, t.bm), E)  ->  PWR.align_path(numel, E)
    return PWR.align_path(n.numel, cfg.num_experts)


# ---- round5_frozen/2026-10-21/diag-A2/work/common.py:55 price_parts
def price_parts(model, arm, c, doc, prm, *, key, reads=None, x_over=None):
    if not isinstance(c["n"], Tread):
        return _orig("price_parts")(model, arm, c, doc, prm, key=key, reads=reads, x_over=x_over)
    STATS["price_parts"] += 1
    cm = _NS(_orig("price_parts"))
    RG, WV4, TM, V4, TV3 = cm.RG, cm.WV4, cm.TM, cm.V4, cm.TV3
    import r6model as M6  # noqa: F401
    pr = RG.pricer()
    t = RG.PAGE_TILE[int(doc["block_m"])]
    # common.py:60  n, counts, G = int(c["n"]), c["counts"], int(doc["group_m"])  ->  n = c["n"] (the Tread)
    n, counts, G = c["n"], c["counts"], int(doc["group_m"])
    seed = RG.seed_of(doc, c)
    base = WV4.price(pr, model, arm, n, counts, RG.COPIES, seed, t, G, prm, key=key, reads=reads)
    occ = {g: prm.occ(t, g) for g in TM.GEMMS}
    phis = {g: V4.phi(t, g, prm) for g in TM.GEMMS}
    scale = {g: V4.c_scale(t, g, prm) for g in TM.GEMMS}
    x = [float(v) for v in pr.f2.x]
    if x_over:
        for i, v in x_over.items():
            x[i] = v
    T0, c_ns, bw, s_small, s_block = x
    ct = None if counts is None else tuple(int(v) for v in counts)
    out = {"ms_v4": base["ms"], "reads": base["reads"], "q": base["q"], "occ": occ, "phi": phis, "scale": scale}
    with V4.at_tile(model, t, phis):
        ctx = pr.ctx if occ == dict(pr.ctx.occupancy) else dataclasses.replace(pr.ctx, occupancy=dict(occ))
        cell = V4.make_cell(pr.eng, model, arm, n, ct, RG.COPIES, G, t, base["reads"], occ, pr.f2.k_w, ctx.sms)
        # common.py:76  treads = cell.n if ct is None else TM.numel_of(cell.n, ct) / TM.IDS_PER_TREAD
        #                (ct is never None here: numel_of reads the counts, not n)
        if ct is None:
            raise PoisonedTread("price_parts: a balanced cell with a Tread")
        treads = TM.numel_of(cell.n, ct) / TM.IDS_PER_TREAD
        out["T0"] = T0
        out["other"] = treads * TM.B_OTHER / (bw * 1e6)
        out["B_other_bytes"] = treads * TM.B_OTHER
        tot = T0 + out["other"]
        with TM.dead_model("v2"):
            for g in TM.GEMMS:
                cg = c_ns * scale[g]
                w = TM.window_width(pr.f2.k_w, ctx.sms, occ[g])
                win = TM.window(arm, cell.declared, G, n, g, w, ct)
                full, fl, by, nt = cm.gemm_parts(win, g, cell.sigma[g], cg, bw, ctx.sms, occ[g])
                dead = TM.dead_ms(cell.declared, n, g, cg, pr.f2.k_w, occ[g], ct)
                const = TM.gemm_const_ms(g, cg)
                out[g] = {"gemm": full, "floor_only": fl, "bytes_only": by, "notail": nt, "dead": dead, "const": const,
                          "live": win.live, "dead_ctas": TM.dead_ctas(cell.declared, n, g, ct),
                          "ksteps": TM.GEOMETRY[g].ksteps, "S_prime": TM.floor_ksteps(g), "c_ns": cg,
                          "waves": win.live / (ctx.sms * occ[g]), "reads": base["reads"][g],
                          "W": TM.BYTE_MODEL[f"W_{g}"], "A_per_tread": TM.BYTE_MODEL[f"operand_per_tile_{g}"],
                          "out_tile": TM.OUT_TILE}
                tot += full + dead + const
        out["s"] = (s_small if cell.path == TM.SMALL_BATCH else s_block) if arm == "native" else 0.0
        out["path"] = cell.path
        tot += out["s"]
        out["b"] = TV3.v3_b_ms(arm, base["native_path"], None if base["A_us"] is None else base["A_us"] * 1e-3,
                               key=cell.key) if arm != "native" else 0.0
        tot += out["b"]
    out["ms_sum"] = tot
    return out


# ---- round5_frozen/2026-10-22/diag-v5/work/d5.py:70 v5_parts
_UNSET = object()


def v5_parts(model, c, doc, prm, *, reads=None, F3=None, P=None, rho=True, sag="board", sag_const=None,
             own_off_us=None, r3=_UNSET, key="v3", sag_scale=1.0):
    d5 = _NS(_orig("d5.v5_parts"))
    if r3 is _UNSET:
        r3 = d5.R3_1410
    if not isinstance(c["n"], Tread):
        return _orig("d5.v5_parts")(model, c, doc, prm, reads=reads, F3=F3, P=P, rho=rho, sag=sag,
                                    sag_const=sag_const, own_off_us=own_off_us, r3=r3, key=key, sag_scale=sag_scale)
    STATS["d5.v5_parts"] += 1
    V, RG = d5.V, d5.RG
    F3 = d5.F3P if F3 is None else F3
    P = d5.P5 if P is None else P
    pr = RG.pricer()
    bm = int(doc["block_m"])
    t = RG.PAGE_TILE[bm]
    # d5.py:80  n, counts, G = int(c["n"]), c["counts"], int(doc["group_m"])  ->  n = c["n"] (the Tread)
    n, counts, G = c["n"], c["counts"], int(doc["group_m"])
    seed = RG.seed_of(doc, c)
    with V.swapped(P[bm], V.RHO_X[bm] if rho else None):
        if reads is None:
            reads, _q = V.f3_reads(pr, model, n, counts, seed, t, G, prm, F3[bm], key)
        base = d5.price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        if sag == "none":
            base["fac"] = {"w1": 1.0, "w2": 1.0}
            return base
        fac = {}
        for g in ("w1", "w2"):
            tf = base[g]["floor_only"]
            if sag == "board":
                A, tau = sag_const or d5.BOARD_A2[bm]
                D = d5.sat(A * sag_scale, tau, tf)
            else:
                D = own_off_us[g] * 1e-3
            if r3:
                D -= d5.sat(r3[0], r3[1], tf)
            fac[g] = 1 + D / tf
        with d5.c_patch(fac):
            out = d5.price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        out["fac"] = fac
        out["tf"] = {g: base[g]["floor_only"] for g in ("w1", "w2")}
        return out


# ---- round5_frozen/2026-10-22/refit4/work/r4.py:204 m4_reads
def m4_reads(model, c, doc, p128=5.75):
    if not isinstance(c["n"], Tread):
        return _orig("r4.m4_reads")(model, c, doc, p128)
    STATS["r4.m4_reads"] += 1
    r4 = _NS(_orig("r4.m4_reads"))
    d5, k5, RG, A2 = r4.d5, r4.k5, r4.RG, r4.A2
    prm = d5.params_for(d5.SEL, model)
    bm = int(doc["block_m"])
    with A2.scoring(r4.RULES):
        with d5.V.swapped(p128 if bm == 128 else k5.P64, d5.V.RHO_X[bm]):
            # r4.py:210  d5.V.f3_reads(RG.pricer(), model, int(c["n"]), ...)  ->  c["n"] (the Tread)
            reads, q = d5.V.f3_reads(RG.pricer(), model, c["n"], c["counts"], RG.seed_of(doc, c), RG.PAGE_TILE[bm],
                                     int(doc["group_m"]), prm, k5.F3_M4[bm], "v3")
    return reads, q


# ---- round5_frozen/2026-10-22/knee-v5/work/k5.py:202 cell_ms
def reads_key(r: dict) -> tuple:
    """k5._READS's key of a v8 row: k5.py:207's (model, page, n, label, BM, G) with n its label, extended by the Tread's
    digest of (counts, ids)."""
    n = r["n"]
    return (r["model"], r.get("page"), n.label, r["label"], int(r["doc"]["block_m"]), int(r["doc"]["group_m"]),
            n.digest)


def k5_cell_ms(r, f, p128, sag128, sag64, sag_curve=None):
    if not isinstance(r["n"], Tread):
        return _orig("k5.cell_ms")(r, f, p128, sag128, sag64, sag_curve)
    STATS["k5.cell_ms"] += 1
    k5 = _NS(_orig("k5.cell_ms"))
    d5, RG, A2 = k5.d5, k5.RG, k5.A2
    prm = d5.params_for(d5.SEL, r["model"])
    bm = int(r["doc"]["block_m"])
    # k5.py:207  key = (r["model"], r.get("page"), r["n"], r["label"], bm, G)  ->  reads_key(r)
    key = reads_key(r)
    with A2.scoring(k5.RULES):
        if key not in k5._READS:
            with d5.V.swapped(p128 if bm == 128 else k5.P64, d5.V.RHO_X[bm]):
                # k5.py:211  int(r["n"])  ->  r["n"] (the Tread)
                k5._READS[key] = d5.V.f3_reads(RG.pricer(), r["model"], r["n"], r["c"]["counts"],
                                               RG.seed_of(r["doc"], r["c"]), RG.PAGE_TILE[bm],
                                               int(r["doc"]["group_m"]), prm, k5.F3_M4[bm], "v3")[0]
        reads = k5._READS[key]
        # v8: the guard's None reads are a refusal here; the frozen code went on to price_parts(reads=None), whose
        # WV4.price returns {ms: None} without 'reads' (KeyError('reads'), v7's latent 64x128 G 1 n >= 6 defect)
        if reads is None:
            return None
        P = {128: p128, 64: k5.P64}
        with k5.form_ctx(f if bm == 128 else None):
            if sag_curve is None:
                return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P, rho=True,
                                   sag_const={128: sag128, 64: sag64}[bm])["ms_sum"]
            base = d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P, rho=True, sag="none")
            own = {g: sag_curve[bm](base[g]["floor_only"]) for g in ("w1", "w2")}
            return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P, rho=True, sag="own",
                               own_off_us=own, r3=d5.R3_1410)["ms_sum"]


#: name -> (where the frozen original lives, the copy)
SWAPS = {
    "Geometry.num_pid_m": ("scripts/wave_split_bytes.py:434", num_pid_m),
    "W.walk": ("scripts/wave_split_bytes.py:632", walk),
    "W.cell_events": ("scripts/wave_split_bytes.py:775", cell_events),
    "Model.domain": ("scripts/wave_split_bytes.py:1231", domain),
    "Model.evaluate": ("scripts/wave_split_bytes.py:1241", evaluate),
    "WV3.tile_content_ids_counts": ("scripts/wave_split_bytes_v3.py:22", tile_content_ids_counts),
    "WV4.reads_of": ("scripts/wave_split_bytes_v4.py:116", reads_of),
    "WV4.price": ("scripts/wave_split_bytes_v4.py:154", wv4_price),
    "V4.make_cell": ("scripts/r3_timing_model_v4.py:397", make_cell),
    "V4.native_path_of": ("scripts/r3_timing_model_v4.py:417", native_path_of),
    "price_parts": ("scripts/round5_frozen/2026-10-21/diag-A2/work/common.py:55", price_parts),
    "d5.v5_parts": ("scripts/round5_frozen/2026-10-22/diag-v5/work/d5.py:70", v5_parts),
    "r4.m4_reads": ("scripts/round5_frozen/2026-10-22/refit4/work/r4.py:204", m4_reads),
    "k5.cell_ms": ("scripts/round5_frozen/2026-10-22/knee-v5/work/k5.py:202", k5_cell_ms),
}

#: EVERY binding of every swapped original, grep-complete (critique M2). Found by grepping scripts/ (frozen work files
#: included) for `from X import f`, `f = X.f` and `_ORIG... = X.f` of each name, and held at install time by a scan of
#: every loaded module (sys.modules and round 5's popped work modules, reached through their attributes) for any
#: attribute that IS the original (`bindings()`): a binding not listed here refuses the install.
#:   Geometry.num_pid_m          wave_split_bytes.Geometry.num_pid_m (a method: every caller reads geom.num_pid_m:
#:                               W.cell_events :780, W.load_card :1199/:1203, r4.cell_events_law r4.py:173,
#:                               r5.cell_events_ls r5.py:144)
#:   W.walk (the original)       r4._ORIG_WALK (round5_frozen/2026-10-22/refit4/work/r4.py:60); wave_split_bytes.walk
#:                               itself holds r4.walk_law (r4.py:184), r5/r6/r7 hold their own walks (r5.walk_ls,
#:                               r6.walk_ls4), none of them the original
#:   W.cell_events (original)    r4._ORIG_CELL_EVENTS (r4.py:61); wave_split_bytes.cell_events holds r7.cell_events5
#:                               (r7.py:70, over r5.cell_events_ls r5.py:155 over r4.cell_events_law r4.py:185);
#:                               r5._R4_CELL_EVENTS (r5.py:136) and r7._R5_CE (r7.py:51) hold those wrappers
#:   Model.domain, .evaluate     wave_split_bytes.Model.domain / .evaluate (methods; WV3.ModelV3 inherits both)
#:   WV3.tile_content_ids_counts wave_split_bytes_v3.tile_content_ids_counts (read as a global by
#:                               content_events_counts, wave_split_bytes_v3.py:67)
#:   WV4.reads_of                wave_split_bytes_v4.reads_of (read as a global by keyed_reads :124-130)
#:   WV4.price                   wave_split_bytes_v4.price (every caller reads WV4.price: common.py:62,
#:                               regtile.price_cell, r3_timing_model_v7.price_r3, r6model; no by-name import)
#:   V4.make_cell                r3_timing_model_v4.make_cell (callers read V4.make_cell: WV4.price :188, common.py:75)
#:   V4.native_path_of           r3_timing_model_v4.native_path_of (WV4.price :189)
#:   price_parts                 common.price_parts (common.py:55, its own module), BOUND BY NAME in
#:                               v5price.py:29, kneefit.py:22, d5.py:23 and bw.py:19 (`from d5 import ... price_parts`)
#:   d5.v5_parts                 d5.v5_parts (callers read d5.v5_parts: k5.py:217-221, r4.py:230, r6.py:182)
#:   r4.m4_reads                 r4.m4_reads (callers read r4.m4_reads: r7.py:140/146, r6.py:143, r4.py:227)
#:   k5.cell_ms                  k5.cell_ms (callers read k5.cell_ms: r6.py:174, r3_timing_model_v7.py:378/383)
SWAP_BINDINGS = {
    "Geometry.num_pid_m": {"wave_split_bytes.Geometry.num_pid_m"},
    "W.walk": {"r4._ORIG_WALK"},
    "W.cell_events": {"r4._ORIG_CELL_EVENTS"},
    "Model.domain": {"wave_split_bytes.Model.domain"},
    "Model.evaluate": {"wave_split_bytes.Model.evaluate"},
    "WV3.tile_content_ids_counts": {"wave_split_bytes_v3.tile_content_ids_counts"},
    "WV4.reads_of": {"wave_split_bytes_v4.reads_of"},
    "WV4.price": {"wave_split_bytes_v4.price"},
    "V4.make_cell": {"r3_timing_model_v4.make_cell"},
    "V4.native_path_of": {"r3_timing_model_v4.native_path_of"},
    "price_parts": {"common.price_parts", "v5price.price_parts", "kneefit.price_parts", "d5.price_parts",
                    "bw.price_parts"},
    "d5.v5_parts": {"d5.v5_parts"},
    "r4.m4_reads": {"r4.m4_reads"},
    "k5.cell_ms": {"k5.cell_ms"},
}


def _originals() -> dict:
    """The frozen original of every swapped name (round 5's chain loaded)."""
    if _ORIG:
        return _ORIG
    C = R5.chain()
    import r3_timing_model_v4 as V4
    import wave_split_bytes_v3 as WV3
    import wave_split_bytes_v4 as WV4
    W = _W()
    r4, k5, d5 = C["r4"], C["k5"], C["d5"]
    got = {"Geometry.num_pid_m": W.Geometry.num_pid_m, "W.walk": r4._ORIG_WALK, "W.cell_events": r4._ORIG_CELL_EVENTS,
           "Model.domain": W.Model.domain, "Model.evaluate": W.Model.evaluate,
           "WV3.tile_content_ids_counts": WV3.tile_content_ids_counts, "WV4.reads_of": WV4.reads_of,
           "WV4.price": WV4.price, "V4.make_cell": V4.make_cell, "V4.native_path_of": V4.native_path_of,
           "price_parts": d5.price_parts, "d5.v5_parts": d5.v5_parts, "r4.m4_reads": r4.m4_reads,
           "k5.cell_ms": k5.cell_ms}
    for name, f in got.items():
        path, line = SWAPS[name][0].rsplit(":", 1)
        tail = path.split("round5_frozen/", 1)[1] if "round5_frozen/" in path else path.split("scripts/", 1)[1]
        # round 5's work files load from the private HOME's code/moe-kernels-work/<tail> (round5_chain.frozen_home)
        assert f.__code__.co_filename.endswith("/" + tail), (name, f.__code__.co_filename, tail)
        assert f.__code__.co_firstlineno == int(line), (name, f.__code__.co_firstlineno, line)
    _ORIG.update(got)
    return _ORIG


def _modules() -> list:
    """Every loaded module: sys.modules plus round 5's popped work modules, reached through module attributes and
    through the module of every function they hold (common is reached only through price_parts' globals)."""
    import warnings
    C = R5.chain()
    seen: dict = {}
    with warnings.catch_warnings():     # torch's deprecated module attributes warn when touched
        warnings.simplefilter("ignore")
        return _walk_modules(C, seen)


def _walk_modules(C, seen) -> list:
    stack = [m for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)]
    stack += [v for v in C.values() if isinstance(v, types.ModuleType)]
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen[id(m)] = m
        for v in list(vars(m).values()):
            if isinstance(v, types.ModuleType) and id(v) not in seen:
                stack.append(v)
    return list(seen.values())


def _scan() -> dict:
    """{swapped name: {'module.attr' or 'module.Class.attr': (holder, attr)}}: every attribute of every loaded module
    (and of every class it defines) that IS the original, plus the defining module's own global when that module is
    popped from sys.modules (common: reached only through price_parts' globals)."""
    import warnings
    orig = _originals()
    want = {id(f): k for k, f in orig.items()}
    found: dict = {k: {} for k in orig}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mods = _modules()
        _scan_into(mods, orig, want, found)
    for k, f in orig.items():
        g = f.__globals__
        if g.get(f.__name__) is f:
            found[k].setdefault(f"{g['__name__']}.{f.__name__}", (g, f.__name__))
    return found


def _scan_into(mods, orig, want, found) -> None:
    for m in mods:
        name = getattr(m, "__name__", "?")
        for a, v in list(vars(m).items()):
            k = want.get(id(v))
            if k is not None and v is orig[k]:
                found[k][f"{name}.{a}"] = (m, a)
            elif isinstance(v, type) and getattr(v, "__module__", None) == name:
                for a2, v2 in list(vars(v).items()):
                    k2 = want.get(id(v2))
                    if k2 is not None and v2 is orig[k2]:
                        found[k2][f"{name}.{a}.{a2}"] = (v, a2)


def bindings() -> dict:
    """{swapped name: sorted bindings of the original} over every loaded module (outside `installed()`)."""
    if _DEPTH[0]:
        raise RuntimeError("bindings() inside installed(): the originals are swapped out")
    return {k: sorted(v) for k, v in _scan().items()}


_PLAN: list = []
_PLAN_KEY: list = [None]


def _modules_key():
    """A cheap fingerprint of the loaded module set (sys.modules' ids plus round 5's work modules): the plan is
    re-scanned whenever it changes, so a module imported after the first install that binds an original is caught."""
    C = R5.chain()
    return hash((tuple(id(m) for m in list(sys.modules.values())),
                 tuple(id(v) for v in C.values() if isinstance(v, types.ModuleType))))


def _plan() -> list:
    """[(holder, attribute, original, copy)] for every binding; refuses a binding SWAP_BINDINGS does not declare.

    RE-SCANNED PER INSTALL (review 1, MINOR): the scan is repeated at every outermost `installed()` whose loaded-module
    fingerprint (`_modules_key`) differs from the last scan's, so a module loaded later cannot keep an unswapped
    binding. SCAN BLIND SPOTS (what `_scan` cannot see, stated rather than hidden): an original held inside a
    container (a list, tuple, dict value or set), a closure cell, a default argument, a functools.partial, an instance
    attribute of an object that is not a module or a class defined in one, or a C-level reference. The grep of
    scripts/ behind SWAP_BINDINGS found none of these forms for any swapped name; the canary and the Tread poison
    (a missed site raises PoisonedTread) are the run-time backstop."""
    key = _modules_key()
    if _PLAN and _PLAN_KEY[0] == key:
        return _PLAN
    orig = _originals()
    found = _scan()
    extra = {k: sorted(set(v) - SWAP_BINDINGS[k]) for k, v in found.items() if set(v) - SWAP_BINDINGS[k]}
    if extra:
        raise Refused(f"v8: bindings of swapped functions that SWAP_BINDINGS does not declare: {extra}")
    missing = {k: sorted(SWAP_BINDINGS[k] - set(v)) for k, v in found.items() if SWAP_BINDINGS[k] - set(v)}
    if missing:
        raise Refused(f"v8: declared bindings not found (the chain changed?): {missing}")
    _PLAN.clear()
    _PLAN.extend((holder, attr, orig[k], SWAPS[k][1]) for k, v in found.items() for holder, attr in v.values())
    _PLAN_INFO.update(bindings={k: sorted(v) for k, v in found.items()}, scans=_PLAN_INFO.get("scans", 0) + 1)
    _PLAN_KEY[0] = key
    return _PLAN


_PLAN_INFO: dict = {}
_DEPTH = [0]


def _set(holder, attr, value):
    if isinstance(holder, dict):
        holder[attr] = value
    else:
        setattr(holder, attr, value)


@contextlib.contextmanager
def installed():
    """Inside this block every binding of every inventoried function is v8's copy; restored on exit (re-entrant)."""
    if _DEPTH[0]:
        _DEPTH[0] += 1
        try:
            yield
        finally:
            _DEPTH[0] -= 1
        return
    plan = _plan()          # outside any install: re-scanned when the loaded modules changed
    done = []
    try:
        for holder, attr, f, copy in plan:
            cur = holder.get(attr) if isinstance(holder, dict) else vars(holder).get(attr)
            if cur is not f and cur is not copy:
                raise Refused(f"v8: {attr} moved under the swap ({cur!r})")
            _set(holder, attr, copy)
            done.append((holder, attr, f))
        _DEPTH[0] = 1
        yield
    finally:
        _DEPTH[0] = 0
        for holder, attr, f in reversed(done):
            _set(holder, attr, f)


# ================================================================== the prefill branch (v7's, at a Tread)

PREFILL_TILES, DECODE_TILES = V7.PREFILL_TILES, V7.DECODE_TILES
#: The prefill branch's three on-board / registered inputs have NO default (review 1, BLOCKER 2): every prefill call
#: names its 128x128 sag par (`sag128`, (A us, tau us), measured on the board at 1410 MHz), its 64x128 par (`sag64`)
#: and its K1 knee (`knee`, (p, S0)); a call without them is a ValueError, never a silent constant from another
#: board. The production kit passes them from scripts/prod/prodlib.py (PREFILL_REGISTERED + the board's par).
PREFILL_KEYS = ("sag128", "sag64", "knee")


def hopper_refused(n: Tread, G: int) -> bool:
    return int(G) == 1 and n.hopper_g1(n.E, n.bm)


def price_prefill(row: dict, *, sag128, sag64, knee, switches: Switches = V8) -> dict:
    """V7.price_prefill (r3_timing_model_v7.py:353) on a row whose n is a Tread, inside `installed()`. The two changed
    lines are marked; a None from the guard (128: STAG-CUT; 64: k5.cell_ms) is a refusal with its reason."""
    C = R5.chain()
    bm, G = int(row["doc"]["block_m"]), int(row["doc"]["group_m"])
    tile = V7.prefill_tile_of(row)
    if tile not in PREFILL_TILES:
        raise Refused(f"prefill cell at BLOCK_M {bm}: no prefill tile")
    n = row["n"]
    assert isinstance(n, Tread), n
    if sag128 is None or sag64 is None or knee is None:
        raise ValueError(f"price_prefill needs sag128, sag64 and knee (got {sag128!r}, {sag64!r}, {knee!r})")
    p, s0 = tuple(knee)
    rr = switches.RR and V7.rr_applies(tile, G)
    dom = (f"HOPPER_G1_N: w2 at G 1 with n_eff {n.n_eff:.4g} >= {_W().HOPPER_G1_N} is ILL-POSED (M4's Hopper rule, "
           "read at n_eff)") if hopper_refused(n, G) else None
    with installed():
        # r3_timing_model_v7.py:365  rr_w1(row["model"], counts, int(row["n"]), bm)  ->  the Tread (cell_counts reads
        # the counts, always given here; n is never read)
        term = V7.rr_w1(row["model"], row["c"]["counts"], n, bm) if rr else None
        if bm == 128:
            rd = R5.reads(row, p)
            if rd is None:
                return {"ms": None, "regime": "prefill", "tile": tile, "G": G, "terms": {},
                        "why": dom or "the STAG-CUT guard refuses the cell (M4's kappa or root-gap guard)"}
            rd = {g: rd[g] for g in ("w1", "w2")}
            frozen_w1 = float(rd["w1"])
            if rr:
                rd["w1"] = term["bytes"]
            ms = R5.price_128(row, sag128, rd, p, s0)
        else:
            # r3_timing_model_v7.py:376  key = (model, page, row["n"], label, bm, G)  ->  reads_key(row)
            key = reads_key(row)
            with C["r5"].law3():
                ms = C["k5"].cell_ms(row, None, R5.P_V5PP, tuple(sag128), tuple(sag64))
                if ms is None:
                    return {"ms": None, "regime": "prefill", "tile": tile, "G": G, "terms": {},
                            "why": dom or "M4's guard refuses the cell (kappa or root gap)",
                            "defect": "v7 raised KeyError('reads') here (64x128 G 1 n >= 6); v8 refuses"}
                rd = dict(C["k5"]._READS[key])
                frozen_w1 = float(rd["w1"])
                if rr:
                    C["k5"]._READS[key] = dict(rd, w1=term["bytes"])
                    ms = C["k5"].cell_ms(row, None, R5.P_V5PP, tuple(sag128), tuple(sag64))
                    rd = dict(C["k5"]._READS[key])
    out = {"ms": float(ms), "regime": "prefill", "tile": tile, "G": G,
           "reads": {g: float(rd[g]) for g in ("w1", "w2")}, "w1_frozen": frozen_w1, "terms": {}}
    if rr:
        out["terms"]["RR"] = {k: v for k, v in term.items()}
    return out


def prefill_row(model: str, counts, bm: int, G: int, *, ids=None, page: dict | None = None,
                order_seed: int | None = None) -> dict:
    """A v8 prefill row (round 5's row form) for one call: n a Tread over the counts; the doc's shuffle seed the
    call's RealIds (IDS), else the page's seed (a page cell with counts; None on a balanced page cell, as
    regtile.seed_of gives), else None (realize_counts unshuffled)."""
    cfg = MODEL_CONFIGS[model]
    page = page or {}
    T = int(sum(counts)) // cfg.top_k
    if ids is not None:
        seed = RealIds(ids, T, cfg.top_k, order_seed)
    else:
        if order_seed is not None:
            raise ValueError("order_seed needs the call's ids")
        seed = page.get("shuffle_seed")
    n = Tread(counts, cfg.num_experts, bm, label=page.get("n"), seed=seed)
    label = page.get("label", "call")
    name = page.get("page", "call")
    doc = {"block_m": int(bm), "group_m": int(G), "shuffle_seed": seed, "page": name, "model": model}
    c = {"n": n, "counts": list(n.counts), "label": label}
    return {"model": model, "page": name, "n": n, "label": label, "c": c, "doc": doc}


# ================================================================== the decode branch (v7's, with real ids)

_DCONST: dict = {}


def decode_constants() -> dict:
    """The v7 decode blind's frozen constants as its scorer builds them: finalise_decode.params_of on the frozen block
    of docs/registered/2026-11-17-v7-decode-blind-gh200.json, its terms (V5D.Terms.from_dict) and law
    (V6D.Law(gate_d6._tuples)), and D6FIX's prm_fix (v7bcommon.prm_fix_of)."""
    if not _DCONST:
        V7.decode_modules()
        d = REPO / "scripts" / "scoring" / "v7blind"
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        import d5common as D5
        import d6common as D6
        import finalise_decode as FN
        import gate_d6 as G
        import v7bcommon as VB
        fz = json.loads(DECODE_REG.read_text())["predictions"]["frozen"]
        m = fz["model"]
        _DCONST.update(fz=fz, prm=FN.params_of(fz), terms=D5.V5D.Terms.from_dict(m["terms"]),
                       law=D6.V6D.Law(G._tuples(m["law"])), prm_fix=VB.prm_fix_of(m["D6FIX"], fz.get("tile_params")))
    return _DCONST


def price_decode_cell(model: str, cell: dict, prm, *, ids=None, arm: str = "native", decl_copies: int | None = None,
                      fixes=None, terms=None, law=None, prm_fix=None, switches: Switches = V8,
                      naive_v4dp: bool = False) -> dict | None:
    """V7.price_decode_cell (r3_timing_model_v7.py:415) with the IDS switch: the naive path's launch order is the
    call's ids when given (and switches.IDS), else V4DP.ids_of (the page cell's generated ids), as v7."""
    X = V7.decode_modules()
    D6, V6D, V4DP, RD = X["D6"], X["V6D"], X["V4DP"], X["RD"]
    fixes = V4DP.ALL if fixes is None else fixes
    terms = D6.registered_terms() if terms is None else terms
    law = V6D.LAW if law is None else law
    cfg = MODEL_CONFIGS[model]
    declared = None if arm == "native" else cfg.num_experts * int(decl_copies or 1)
    naive = RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == RD.NAIVE
    # r3_timing_model_v7.py:430  ids = V4DP.ids_of(model, cell) if fixes.naive_order and naive else None
    #                       ->  the call's ids (token-major, T k) when given, else V4DP.ids_of
    if fixes.naive_order and naive:
        if ids is not None and switches.IDS:
            STATS["decode.naive_ids"] += 1        # the call's ids reached the naive path's launch order
            ids = tuple(int(x) for x in np.asarray(ids).reshape(-1))
        else:
            ids = V4DP.ids_of(model, cell)
    else:
        ids = None

    def one(prm_):
        try:
            p = V6D.price(model, int(cell["T"]), cell["counts"], prm_, ids=ids, fixes=fixes, terms=terms,
                          declared=declared, law=None if (naive_v4dp and naive) else law)
        except V4DP.TM.Refused:
            return None
        if prm_.T0_GR_ms is not None and p["ms"] is None:
            return None
        return p
    p = one(prm)
    if (p is not None and switches.D6FIX and not naive_v4dp and prm_fix is not None and prm.T0_GR_ms is not None
            and V7.v4dp_branch(p)):
        q = one(prm_fix)
        if q is not None:
            assert V7.v4dp_branch(q), "the D6-24 branch moved with the CAL constants"
            q["v7"] = {"D6FIX": {"rho_form": prm_fix.rho_form, "s_naive_GR_ms": prm_fix.s_naive_GR_ms,
                                 "T0_GR_ms": prm_fix.T0_GR_ms, "v6d_ms": p["ms"]}}
        return q
    return p


# ================================================================== one entry point

def tile_key(cfg: dict) -> str:
    return (f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']}w{cfg['num_warps']}"
            f"s{cfg['num_stages']}")


def default_tile(T: int, E: int) -> tuple[str, int]:
    """vLLM 0.27.1's default ladder (moe.bench.tile_resolve.default_config(T, E)): (tile key, GROUP_SIZE_M)."""
    from moe.bench.tile_resolve import default_config
    d = default_config(int(T), int(E))
    return tile_key(d), int(d["GROUP_SIZE_M"])


#: EVERY reason v8 may give for a refusal (ms None), as (class, the why's prefix): the declared domain edges, the frozen
#: guards and the refusals the frozen code raises as a Refused (review 1, MAJOR 11: a sweep asserts every refusal it
#: meets is one of these). Any other exception is raised, never a refusal.
DECLARED_REFUSALS = (
    ("HOPPER_G1_N", "HOPPER_G1_N:"),                         # w2 at G 1, n_eff >= 6 (M4's Hopper rule, read at n_eff)
    ("STAG_CUT_GUARD", "the STAG-CUT guard refuses the cell"),  # 128x128: M4's kappa or root-gap guard
    ("M4_GUARD", "M4's guard refuses the cell"),              # 64x128: the same guard (k5.cell_ms None)
    ("NUMEL_0", "numel 0: no routed row"),
    ("GEOM_OFF", "GEOM off: v7 prices whole page treads only"),
    ("V6D_NOT_PRICED", "NOT PRICED (v6d:"),                   # decode: no constants for the tile or path
    ("V6D_OTHER_TILE", "v6d prices this cell at "),           # decode: v6d's own tile differs from the call's
    ("REFUSED", "Refused: "),                                 # a frozen or v8 Refused (r3 tile, arm, ids/counts)
)


def refusal_class(why: str | None) -> str | None:
    """The DECLARED_REFUSALS class of a refusal's why, else None (an undeclared refusal)."""
    for cls, pre in DECLARED_REFUSALS:
        if why and why.startswith(pre):
            return cls
    return None


#: what a refusal may be (a declared domain edge); anything else is raised
def _refusals():
    import cross_model_predict as CMP
    import r3_timing_model as TM
    return (V7.Refused, R5.Refused, TM.Refused, _W().Refused, CMP.Refused)


def _model_of(call: dict) -> str:
    m = call["model"]
    if isinstance(m, MoEConfig):
        return register_model(m)
    if m not in MODEL_CONFIGS:
        raise Refused(f"model {m!r} is not in MODEL_CONFIGS; pass its MoEConfig")
    return m


#: a long pricing worker clears the per-call caches every this many prefill calls (review 1, MINOR: bounded memory)
CACHE_EVERY = 256
_CALLS = [0]


def clear_caches() -> dict:
    """Drop the per-call memo entries a stream of distinct real calls would grow without bound: v8's real-ids tile
    contents (_CONTENT_REAL: each key holds the call's ids, 0.8 MB at T 16384; measured 2026-11-19 as the one cache
    that grows per distinct call), round 3's launch-start draws (r5._STARTS, keyed by live CTAs), and, should a path
    leave them, k5._READS entries keyed by a Tread digest (v8's 7-element keys; v7's 6-element page keys stay), WV4
    model event entries keyed by a Tread, and r3_timing_model's functools caches of schedule() and window() (round 5's
    reads5 context already clears k5._READS and WV4._MODELS per call). Every entry is a pure function of its key
    (seeded draws included), so a cleared entry recomputes to the same bits. Returns how many entries each dropped."""
    out = {"_CONTENT_REAL": len(_CONTENT_REAL)}
    _CONTENT_REAL.clear()
    C = R5.chain() if R5._CHAIN else {}
    r5 = C.get("r5")
    if r5 is not None and isinstance(getattr(r5, "_STARTS", None), dict):
        out["r5._STARTS"] = len(r5._STARTS)
        r5._STARTS.clear()
    k5 = C.get("k5")
    if k5 is not None:
        drop = [k for k in k5._READS if isinstance(k, tuple) and len(k) == 7]
        for k in drop:
            del k5._READS[k]
        out["k5._READS"] = len(drop)
    WV4 = sys.modules.get("wave_split_bytes_v4")
    n_ev = 0
    if WV4 is not None:
        for m in list(WV4._MODELS.values()):
            ev = getattr(m, "_events", None)
            if ev:
                drop = [k for k in ev if any(isinstance(x, (Tread, RealIds)) for x in k)]
                for k in drop:
                    del ev[k]
                n_ev += len(drop)
    out["events"] = n_ev
    TM = sys.modules.get("r3_timing_model")
    if TM is not None:
        for f in ("schedule", "window"):
            fn = getattr(TM, f, None)
            if hasattr(fn, "cache_clear"):
                out[f"TM.{f}"] = fn.cache_info().currsize
                fn.cache_clear()
    return out


def prefill_constants(call: dict) -> tuple:
    """(sag128, sag64, knee) of a prefill call, each a pair; a missing one is a ValueError (no default)."""
    miss = [k for k in PREFILL_KEYS if call.get(k) is None]
    if miss:
        raise ValueError(f"a prefill call needs {', '.join(PREFILL_KEYS)}: {', '.join(miss)} missing (v8 has no "
                         "default constants; the production kit passes the board's par and the registered pair)")
    out = []
    for k in PREFILL_KEYS:
        v = tuple(call[k])          # as given (no float(): the frozen code sees the caller's numbers unchanged)
        if len(v) != 2:
            raise ValueError(f"{k} {call[k]!r}: a pair")
        out.append(v)
    return tuple(out)


def price(call: dict, switches: Switches = V8) -> dict:
    """v8's time of one fused-MoE call (see the module docstring for the call). Returns {ms, regime, tile, G, numel,
    n_eff, terms, ...}; a declared refusal returns ms None with `why`. A call {tile: R3's tile, r3: <v7's r3 cell>}
    is v7's r3 branch (R3's registered predictions), priced inside the swaps."""
    tile = call.get("tile")
    if tile == V7.R3_TILE:
        with installed():
            return V7.price_r3(call["r3"])
    model = _model_of(call)
    cfg = MODEL_CONFIGS[model]
    T = int(call["T"])
    counts = tuple(int(v) for v in call["counts"])
    if len(counts) != cfg.num_experts:
        raise ValueError(f"{len(counts)} counts on {model}'s E {cfg.num_experts}")
    numel = sum(counts)
    # the integrity guard (critique MINOR): sum(counts) == T_pad x k, kept on every call
    if numel != T * cfg.top_k:
        raise ValueError(f"sum(counts) {numel} != T {T} x k {cfg.top_k}")
    if numel < 1:
        return {"ms": None, "regime": None, "why": "numel 0: no routed row", "numel": 0}
    dt, dG = default_tile(T, cfg.num_experts)
    tile = tile or dt
    G = int(call.get("G", dG if tile == dt else 1))
    ids = call.get("ids") if switches.IDS else None
    base = {"tile": tile, "G": G, "numel": numel, "T": T, "model": model}
    try:
        reg = V7.regime_of(tile)
        if reg == "prefill":
            bm = int(tile.split("x")[0])
            row = prefill_row(model, counts, bm, G, ids=ids, page=call.get("page"),
                              order_seed=call.get("order_seed"))
            n = row["n"]
            base.update(n_eff=n.n_eff)
            sag128, sag64, knee = prefill_constants(call)
            _CALLS[0] += 1
            if CACHE_EVERY and _CALLS[0] % CACHE_EVERY == 0:
                clear_caches()
            if not switches.GEOM:
                if not n.whole() or ids is not None:
                    return dict(base, ms=None, regime="prefill", why="GEOM off: v7 prices whole page treads only")
                vrow = dict(row, n=n.numel // (n.E * n.bm), c=dict(row["c"], n=n.numel // (n.E * n.bm)),
                            doc=dict(row["doc"], shuffle_seed=row["doc"]["shuffle_seed"]))
                p = V7.price_prefill(vrow, sag128=sag128, sag64=sag64, knee=knee, switches=switches.v7())
            else:
                p = price_prefill(row, sag128=sag128, sag64=sag64, knee=knee, switches=switches)
            return dict(base, **p)
        if reg == "decode":
            dc = decode_constants()
            d = dict(call.get("decode") or {})
            page = call.get("page") or {}
            cell = {"T": T, "counts": list(counts), "label": page.get("label", "call"),
                    "shuffle": bool(page.get("shuffle", True))}
            p = price_decode_cell(model, cell, d.get("prm", dc["prm"]), ids=ids, terms=d.get("terms", dc["terms"]),
                                  law=d.get("law", dc["law"]), prm_fix=d.get("prm_fix", dc["prm_fix"]),
                                  switches=switches)
            if p is None:
                return dict(base, ms=None, regime="decode",
                            why="NOT PRICED (v6d: no constants for the cell's tile or path, or its offset unfitted)")
            t = (p.get("v6d") or {}).get("tile") or (p.get("v5d") or {}).get("tile")
            if t is not None and t != tile:
                return dict(base, ms=None, regime="decode", why=f"v6d prices this cell at {t}, not the call's {tile}")
            return dict(base, ms=p["ms"], regime="decode", tile=t or tile,
                        terms={"D6FIX": p["v7"]["D6FIX"]} if "v7" in p else {}, parts=p)
        raise Refused(f"tile {tile}: the r3 tile is priced only from a v7 r3 cell")
    except _refusals() as ex:
        return dict(base, ms=None, regime=None, why=f"{type(ex).__name__}: {ex}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bindings", action="store_true", help="print every binding of every swapped function")
    a = ap.parse_args(argv)
    if not a.bindings:
        ap.print_help()
        return 2
    _plan()
    print(json.dumps(_PLAN_INFO, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
