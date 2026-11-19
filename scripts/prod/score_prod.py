"""The production check's scorer (KIT-SPEC 7). DESCRIPTIVE: numbers, no bars, no verdicts.

    python scripts/prod/score_prod.py \\
        --run A1710=RUN_A1710 --run A1410=RUN_A1410 --run Adefault=RUN_ADEF --run Aon=RUN_AON --run Beager=RUN_B \\
        --replay decode=REPLAY_DECODE_DIR --replay prefill=REPLAY_PREFILL_DIR \\
        --frontier RESULTS/frontier/predictions --sag RESULTS/sag/par.json --model ernie-4.5-21b-a3b --out SCORE_DIR

RUNS ON THE LAPTOP after the rental (review 1, MAJOR 7): scripts/prod/offline.sh fetches nothing itself; it runs
Frontier's predict and this scorer on the pushed results branch. --workers defaults to 4 (an 18 GB laptop).

INPUTS. Each --run LABEL=DIR is a serve_capture directory after nsys_extract (calls.jsonl, counts.npy, ring/).
The locked runs are labelled by their lock (A1710: score T <= 128 calls, the decode branch; A1410: T > 128, the
prefill branch; CRITIQUE B1); a run at default clocks is a PRODUCTION DELTA only, never model error. --replay
BRANCH=DIR is replay.py's run directory (cells.csv), keyed by the sample's labels g<index> of the run the sample
was drawn from (--sample-run, default A1710 for decode and A1410 for prefill as main.sh draws them).

PREDICTORS. v8 (prodlib.price_v8, the one binding; priced in a process pool, cached by (T, counts digest, ids
digest, par)) at the BOARD's 128x128 sag par (--sag: main.sh's sag1410 step, par.json; required) and the
registered 64x128 par and knee (prodlib.PREFILL_REGISTERED): `v8_ms`. Beside it, DESCRIPTIVE, v8 with no on-board
calibration: the prefill calls at the registered reference pars' median (prodlib.REF_PAR_MEDIAN): `v8_refpar_ms`
(decode calls do not read the par: the same number). The ROOFLINE (required, never NOT SCORED): the v7 decode
blind's registered rival T0_GR + A W_e / bw at its frozen constants (prodlib.roofline_ms), every call: `roof_ms`.
Frontier's predictions (frontier_op.py), each arm its own predictor (`frontier:<arm>`), every one scored against
the SAME serving kernel-sum, labelled by FRONTIER_ARMS: `cli` (complete grid, trained on the kernel-sum WITHOUT the
functional API's fills: the like-for-like truth), `simpath`, `cli-fills` (trained on Frontier's NATIVE definition,
every kernel of the functional call, the fills included: what Frontier itself would train on), `cli-partial` (only
when grid points were missing), `cli-balanced` (opt-in round-robin routing).

TRUTHS. kernel-sum (align..moe_sum durations) is the PRIMARY truth; span (first start to last end) is SECONDARY: a
call with T > 512 runs outside CUDA graphs (the capture sizes stop at 512), so its span includes the host's launch
gaps (the smoke: 128x128 span ~2x kernel-sum); the report gives span / kernel-sum per run by cudagraph mode. The
replay's per-call median (clean repeats) for the sample. Off-lock calls (per-call SM clock more than 15 MHz from the
run's lock) are dropped and counted.
FILLS (SMOKE-FIXES 6): the replay times vLLM's functional fused_experts, which launches ~9 fill kernels per call that
serving's modular TritonExperts does not; --replay-check BRANCH=replay_check.json (nsys_extract replay-check on the
replay under nsys at the branch clock) gives each sampled cell's fills' GPU time, and the replay-vs-serving delta is
reported with and without them (replay_ms minus the cell's fills). v7/v8's decode constants were calibrated on the
functional API (fills inside T0 / align): expect v8 to over-predict in serving by about the fills' time.

REPORTED per (predictor, truth, group): n, MAPE, wMAPE (time-weighted), p50 and p95 of |e|, bias (mean signed e),
a 95% bootstrap CI of MAPE and wMAPE resampling STEPS (calls of one step move together); groups: all, per tile,
per branch, per stratum (64x128 band, small-batch align on a prefill tile, the 8192..8255 refused window), and the
per-layer-per-step total (a step's calls summed). Plus: v8's refused calls' share of MoE time; run deltas
(in-serving vs replay on the same calls; shared stream on vs off, default vs lock, eager vs graphs: per-tile
medians of kernel-sum).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402

OFF_LOCK_MHZ = 15.0
BOOT = 400
BOOT_SEED = 20261119


# ------------------------------------------------------------------------------------------------------------
# metrics (pure)
# ------------------------------------------------------------------------------------------------------------

def metrics(pred, truth) -> dict:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    m = np.isfinite(p) & np.isfinite(t) & (t > 0)
    p, t = p[m], t[m]
    if not len(t):
        return {"n": 0}
    e = (p - t) / t
    ae = np.abs(e)
    return {"n": int(len(t)), "MAPE": float(100 * ae.mean()), "wMAPE": float(100 * np.abs(p - t).sum() / t.sum()),
            "p50_abs_e": float(100 * np.percentile(ae, 50)), "p95_abs_e": float(100 * np.percentile(ae, 95)),
            "bias": float(100 * e.mean()), "truth_ms_sum": float(t.sum())}


def bootstrap_by_step(pred, truth, steps, *, reps: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """95% CI of MAPE and wMAPE, resampling whole steps."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    keys_in = [k if not isinstance(k, list) else tuple(k) for k in steps]
    m = np.isfinite(p) & np.isfinite(t) & (t > 0)
    p, t = p[m], t[m]
    s = [k for k, keep in zip(keys_in, m.tolist(), strict=True) if keep]
    if len(t) < 2:
        return {}
    groups: dict = defaultdict(list)
    for i, k in enumerate(s):
        groups[k].append(i)
    keys = list(groups)
    idx = [np.asarray(groups[k]) for k in keys]
    rng = random.Random(seed)
    mape, wmape = [], []
    for _ in range(reps):
        pick = np.concatenate([idx[rng.randrange(len(keys))] for _ in keys])
        pp, tt = p[pick], t[pick]
        mape.append(100 * np.mean(np.abs(pp - tt) / tt))
        wmape.append(100 * np.abs(pp - tt).sum() / tt.sum())
    q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]  # noqa: E731
    return {"MAPE_ci95": q(mape), "wMAPE_ci95": q(wmape), "reps": reps, "steps": len(keys)}


def step_totals(rows: list[dict], pred_key: str, truth_key: str) -> tuple[list, list, list]:
    """Per step: the sum over its calls (every layer) of pred and of truth; a step with any missing value is
    left out (counted by the caller)."""
    acc: dict = defaultdict(lambda: [0.0, 0.0, 0, True])
    for r in rows:
        a = acc[(r.get("_run"), r["step"])]
        pv, tv = r.get(pred_key), r.get(truth_key)
        if pv is None or tv is None:
            a[3] = False
            continue
        a[0] += pv
        a[1] += tv
        a[2] += 1
    keys = [k for k, v in acc.items() if v[3] and v[2]]
    return [acc[k][0] for k in keys], [acc[k][1] for k in keys], keys


def off_lock(sm_mhz, lock) -> bool:
    return lock is not None and sm_mhz is not None and abs(float(sm_mhz) - float(lock)) > OFF_LOCK_MHZ


# ------------------------------------------------------------------------------------------------------------
# pricing (process pool, cached)
# ------------------------------------------------------------------------------------------------------------

def _price_one(job):
    """One pricing job in a worker. EVERY exception is caught and recorded as the job's error, BaseException
    included (v8's PoisonedTread is a BaseException by design, so no `except Exception` in the frozen code can
    swallow it: review 1, MINOR); only a KeyboardInterrupt propagates."""
    model_key, T, counts, ids, sag128, module = job
    try:
        ans = PL.price_v8(PL.MODELS[model_key].cfg, T, counts, ids, sag128=sag128, module=module)
        return {"ms": ans["ms"], "tile": ans.get("tile"), "regime": ans.get("regime"), "why": ans.get("why", "")}
    except KeyboardInterrupt:
        raise
    except BaseException as exc:                                    # noqa: BLE001
        return {"ms": None, "error": f"{type(exc).__name__}: {exc}"}


def price_all(model_key: str, jobs: dict, *, module: str | None, workers: int) -> dict:
    """jobs: cache key -> (T, counts, ids, sag128). Returns key -> answer. An error (not a refusal) is kept as
    error."""
    keys = list(jobs)
    args = [(model_key, jobs[k][0], jobs[k][1], jobs[k][2], jobs[k][3], module) for k in keys]
    if workers <= 1:
        res = [_price_one(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_price_one, args, chunksize=64))
    return dict(zip(keys, res, strict=True))


# ------------------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------------------

def load_run(label: str, d, model: PL.ServedModel, *, want_ids: bool) -> list[dict]:
    d = Path(d)
    rows = PL.read_calls(d)
    if not rows:
        raise PL.KitError(f"{d}: no calls (calls.jsonl, its shards or its .xz)")
    counts = PL.load_counts(d)
    if len(rows) != len(counts):
        raise PL.KitError(f"{d}: {len(rows)} calls, {len(counts)} count rows")
    ids = {}
    if want_ids:
        recs = PL.decode_shards(sorted((d / "ring").glob("shard_*.npz")), model.cfg.top_k)
        ids = {r["g"]: r["ids"] for r in recs}
    for r, c in zip(rows, counts, strict=True):
        r["_run"] = label
        r["counts"] = [int(x) for x in c]
        r["ids"] = ids.get(r["g"])
        r["kernel_sum_ms"] = None if r.get("kernel_sum_ns") is None else r["kernel_sum_ns"] / 1e6
        r["span_ms"] = None if r.get("span_ns") is None else r["span_ns"] / 1e6
    return rows


def lock_of(label: str):
    for lk in (1710, 1410):
        if str(lk) in label:
            return lk
    return None


def find_replay_csv(d, branch: str) -> Path:
    """replay.py's cells.csv for one branch under d: d itself, or the unique private_weight_reference/<run id>
    whose name carries 'prodreplay' and '-<branch>-' (locked_r3 names the run directory, replay.py the run id)."""
    d = Path(d)
    if (d / "cells.csv").exists():
        return d / "cells.csv"
    hits = [p for p in d.rglob("cells.csv") if "prodreplay" in p.parent.name and f"-{branch}-" in p.parent.name]
    if len(hits) != 1:
        raise PL.KitError(f"{d}: {len(hits)} prod replay runs for branch {branch} (want 1): {[str(h) for h in hits][:3]}")
    return hits[0]


def load_replay(d, branch: str = "") -> dict:
    """{label g<index>: median of clean ms_p50 across repeats} from replay.py's cells.csv."""
    import csv
    out: dict = defaultdict(list)
    with open(find_replay_csv(d, branch), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok" or r.get("clock_drift_ok") in ("False", "false"):
                continue
            out[r["label"]].append(float(r["ms_p50"]))
    return {k: float(np.median(v)) for k, v in out.items() if v}


def load_replay_check(path) -> dict:
    """{label: fills ms per call (the replay's median)} from nsys_extract's replay_check.json; {} when absent."""
    p = Path(path)
    if not p.is_file():
        return {}
    doc = json.loads(p.read_text())
    out = {}
    for c in doc.get("cells", []):
        f = (c.get("replay") or {}).get("fills_ns")
        if f is not None:
            out[c["label"]] = float(f) / 1e6
    return out


def span_ratio(rows: list[dict]) -> dict:
    """median span / kernel-sum by cudagraph mode (eager calls carry the host's launch gaps in their span)."""
    acc: dict = defaultdict(list)
    for r in rows:
        if r.get("kernel_sum_ms") and r.get("span_ms"):
            acc[str(r.get("cg_mode"))].append(r["span_ms"] / r["kernel_sum_ms"])
    return {k: {"n": len(v), "median": float(np.median(v))} for k, v in sorted(acc.items())}


#: what each Frontier arm is (frontier_op.py module doc "TWO TRUTH DEFINITIONS", "ARMS"); the report prints these
FRONTIER_ARMS = {
    "cli": "Frontier's CLI trainer on the complete grid, uniform routing x 3 samples; trained on the kernel-sum WITHOUT "
           "the functional API's fills: the like-for-like truth (serving's kernel-sum has no such fills)",
    "simpath": "as cli, the simulator's features (total_routed_tokens, num_tokens_post_padded)",
    "cli-fills": "as cli, trained on Frontier's NATIVE definition (record_function: every kernel of the functional "
                 "call, its ~9 fills included); scored against the fills-free serving kernel-sum, so it carries the "
                 "fills as error by construction",
    "cli-partial": "as cli on an INCOMPLETE grid (csv_meta.json lists the missing points): secondary, never the primary",
    "cli-balanced": "as cli, Frontier's round-robin (balanced) routing: the opt-in second arm",
}


def load_frontier(path) -> dict:
    """{(run, g, arm): pred_ms} from frontier_op.py predict's directory (index.json + npy), or a predictions.jsonl."""
    if Path(path).is_dir():
        import frontier_op as FO
        return FO.load_predictions(path)
    out = {}
    for r in PL.read_jsonl(path):
        if r.get("pred_ms") is not None:
            out[(r.get("run"), int(r["g"]), r["arm"])] = float(r["pred_ms"])
    return out


# ------------------------------------------------------------------------------------------------------------
# the score
# ------------------------------------------------------------------------------------------------------------

def table(rows: list[dict], pred_key: str, truth_key: str, *, with_ci: bool = True) -> dict:
    """metrics for all / per tile / per branch / per stratum, plus the per-step total."""
    def sel(f):
        rr = [r for r in rows if f(r) and r.get(pred_key) is not None and r.get(truth_key) is not None]
        m = metrics([r[pred_key] for r in rr], [r[truth_key] for r in rr])
        if with_ci and m.get("n", 0) > 1:
            m.update(bootstrap_by_step([r[pred_key] for r in rr], [r[truth_key] for r in rr],
                                       [(r.get("_run"), r.get("step")) for r in rr]))
        return m
    out = {"all": sel(lambda r: True), "tile": {}, "branch": {}, "stratum": {}}
    for lab, _lo, _hi in PL.TILE_BINS:
        m = sel(lambda r, lab=lab: r["tile"] == lab)
        if m.get("n"):
            out["tile"][lab] = m
    for b in ("decode", "prefill"):
        out["branch"][b] = sel(lambda r, b=b: r["branch"] == b)
    for s in ("band_64x128", "small_batch_align_on_prefill_tile", PL.REFUSED_STRATUM):
        m = sel(lambda r, s=s: s in (r.get("strata") or []))
        if m.get("n"):
            out["stratum"][s] = m
    ps, ts, _k = step_totals([r for r in rows if r.get("step") is not None], pred_key, truth_key)
    out["per_step_total"] = metrics(ps, ts)
    return out


def tile_medians(rows: list[dict], key: str) -> dict:
    acc: dict = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            acc[r["tile"]].append(r[key])
    return {k: {"n": len(v), "median_ms": float(np.median(v))} for k, v in acc.items()}


def delta(a: dict, b: dict) -> dict:
    return {k: {"a_ms": a[k]["median_ms"], "b_ms": b[k]["median_ms"],
                "delta_pct": 100 * (a[k]["median_ms"] - b[k]["median_ms"]) / b[k]["median_ms"]}
            for k in a if k in b and b[k]["median_ms"] > 0}


def score(*, model: PL.ServedModel, runs: dict, replay: dict, sample_run: dict, frontier: dict, roof: dict,
          prices: dict, sag128=None, fills: dict | None = None) -> dict:
    """runs: label -> rows (load_run, priced: r['v8_ms'] etc. filled by the caller); replay: branch -> {label: ms}."""
    out: dict = {"model": model.key, "descriptive": True, "runs": {}, "deltas": {}, "notes": [],
                 "constants": {"sag128 (board, measured at 1410)": None if sag128 is None else list(sag128),
                               "sag64, knee (registered)": {k: PL.PREFILL_REGISTERED[k] for k in ("sag64", "knee")},
                               "sources": PL.PREFILL_REGISTERED["source"],
                               "v8_refpar_ms (DESCRIPTIVE, no on-board calibration)": list(PL.REF_PAR_MEDIAN),
                               "roofline": roof}}
    out["notes"].append("v8_refpar_ms is DESCRIPTIVE: v8 with no on-board calibration (the registered reference "
                        "pars' median for sag128); v8_ms at the board's par is the model's number")
    out["notes"].append("truths: kernel_sum_ms is PRIMARY; span_ms is SECONDARY (calls with T > 512 run without CUDA "
                        "graphs and their span includes host launch gaps: the smoke's 128x128 span was ~2x kernel-sum; "
                        "span / kernel-sum per cudagraph mode under each run)")
    fills = fills or {}
    arms = sorted({a for (_r, _g, a) in frontier})
    if arms:
        out["frontier_arms"] = {a: FRONTIER_ARMS.get(a, "unlabelled arm") for a in arms}
        out["notes"].append("Frontier arms, all scored against serving's kernel-sum (fills-free): "
                            + "; ".join(f"frontier:{a} = {FRONTIER_ARMS.get(a, 'unlabelled arm')}" for a in arms))
    preds = (["v8_ms", "v8_refpar_ms", "roof_ms"] + [f"frontier:{a}" for a in arms])
    for label, rows in runs.items():
        lk = lock_of(label)
        dropped = sum(1 for r in rows if off_lock(r.get("sm_mhz"), lk))
        keep = [r for r in rows if r.get("status") == "ok" and not r.get("warmup") and not off_lock(r.get("sm_mhz"), lk)]
        if lk == 1710:
            keep = [r for r in keep if r["branch"] == "decode"]
        elif lk == 1410:
            keep = [r for r in keep if r["branch"] == "prefill"]
        res = {"lock": lk, "role": "model+context error at the branch clock" if lk else
               "production delta only (default clocks: never model error)",
               "calls": len(rows), "scored": len(keep), "off_lock_dropped": dropped, "tables": {}}
        tot = sum(r["kernel_sum_ms"] for r in keep if r.get("kernel_sum_ms"))
        ref = sum(r["kernel_sum_ms"] for r in keep if r.get("kernel_sum_ms") and r.get("v8_ms") is None)
        res["v8_refused_share_of_moe_time"] = (ref / tot) if tot else None
        res["v8_refusals"] = dict(sorted(_count([r.get("v8_why") for r in keep if r.get("v8_ms") is None]).items()))
        for p in preds:
            for truth in ("kernel_sum_ms", "span_ms"):
                res["tables"][f"{p} vs {truth}"] = table(keep, p, truth)
        res["intruder_calls"] = sum(1 for r in keep if r.get("intruders"))
        res["span_over_kernel_sum_by_cg_mode"] = span_ratio(keep)
        out["runs"][label] = res
    # the model error proper: v8 (and rivals) vs replay at the branch clock, same calls
    for b, rep in replay.items():
        rows = [r for r in runs.get(sample_run.get(b, ""), []) if f"g{r['g']}" in rep]
        for r in rows:
            r["replay_ms"] = rep[f"g{r['g']}"]
        out["runs"][f"replay-{b}"] = {"lock": PL.LOCKS[b], "sample_run": sample_run.get(b), "calls": len(rows),
                                      "tables": {f"{p} vs replay_ms": table(rows, p, "replay_ms") for p in preds}}
        out["runs"][f"replay-{b}"]["tables"]["kernel_sum_ms (in-serving) vs replay_ms"] = table(rows, "kernel_sum_ms",
                                                                                               "replay_ms")
        fb = fills.get(b) or {}
        if fb:
            for r in rows:
                f = fb.get(f"g{r['g']}")
                r["replay_ms_nofills"] = None if f is None else r["replay_ms"] - f
                r["fills_ms"] = f
            tabs = out["runs"][f"replay-{b}"]["tables"]
            tabs["kernel_sum_ms (in-serving) vs replay_ms minus the API fills"] = table(rows, "kernel_sum_ms",
                                                                                      "replay_ms_nofills")
            tabs["v8_ms vs replay_ms minus the API fills"] = table(rows, "v8_ms", "replay_ms_nofills")
            per: dict = defaultdict(list)
            for r in rows:
                if r.get("fills_ms") is not None:
                    per[r["tile"]].append((r["fills_ms"], r.get("kernel_sum_ms"), r.get("v8_ms")))
            out["runs"][f"replay-{b}"]["fills"] = {
                t: {"n": len(v), "fills_us_median": 1e3 * float(np.median([x[0] for x in v])),
                    "share_of_in_serving_kernel_sum": float(np.median([x[0] / x[1] for x in v if x[1]])) if any(
                        x[1] for x in v) else None,
                    "share_of_v8": float(np.median([x[0] / x[2] for x in v if x[2]])) if any(x[2] for x in v) else None}
                for t, v in per.items()}
    med = {lab: tile_medians(rows, "kernel_sum_ms") for lab, rows in runs.items()}
    pairs = [("Aon", "A1710", "shared-experts stream on vs off (overlap)"),
             ("Adefault", "A1710", "default clocks vs the 1710 lock (clock)"),
             ("Adefault", "A1410", "default clocks vs the 1410 lock (clock)"),
             ("Beager", "Adefault", "eager (mode B) vs graphs (mode A), default clocks")]
    for a, b, what in pairs:
        if a in med and b in med:
            out["deltas"][f"{a} vs {b}"] = {"what": what, "per_tile_median_kernel_sum": delta(med[a], med[b])}
    out["pricing"] = {"distinct_priced": len(prices),
                      "errors": sum(1 for v in prices.values() if v.get("error")),
                      "first_errors": [v["error"] for v in prices.values() if v.get("error")][:5]}
    return out


def _count(xs) -> dict:
    d: dict = defaultdict(int)
    for x in xs:
        d[str(x)] += 1
    return d


def attach_predictions(model: PL.ServedModel, runs: dict, frontier: dict, roof: dict, *, sag128,
                       module: str | None, workers: int) -> dict:
    """v8 at the board's par (v8_ms), v8 at the reference median (v8_refpar_ms, prefill only; decode reads no par),
    the roofline (roof_ms) and Frontier's predictions, onto every row."""
    if sag128 is None:
        raise PL.KitError("no board par (--sag): v8's prefill branch has no default 128x128 sag par")
    jobs = {}
    for rows in runs.values():
        for r in rows:
            base = (r["T_pad"], r["counts_digest"], r["ids_digest"])
            pre = r["T_pad"] > PL.DECODE_MAX_T
            r["_pkey"] = base + ((tuple(sag128),) if pre else (None,))
            r["_rkey"] = base + ((PL.REF_PAR_MEDIAN,) if pre else (None,))
            for k, par in ((r["_pkey"], sag128), (r["_rkey"], PL.REF_PAR_MEDIAN)):
                if k not in jobs:
                    jobs[k] = (r["T_pad"], r["counts"], r["ids"], tuple(par) if pre else None)
    prices = price_all(model.key, jobs, module=module, workers=workers)
    for label, rows in runs.items():
        for r in rows:
            a = prices[r["_pkey"]]
            r["v8_ms"] = a.get("ms")
            r["v8_why"] = a.get("why") or a.get("error")
            r["v8_refpar_ms"] = prices[r["_rkey"]].get("ms")
            r["roof_ms"] = PL.roofline_ms(model.cfg, r["counts"], roof)
        fr: dict = defaultdict(dict)
        for (run, g, arm), v in frontier.items():
            if run in (label, None):
                fr[g][arm] = v
        for r in rows:
            for arm, v in fr.get(r["g"], {}).items():
                r[f"frontier:{arm}"] = v
    return prices


def render(sc: dict) -> str:
    lines = [f"PRODUCTION CHECK (DESCRIPTIVE: no bars, no verdicts) model {sc['model']}", ""]
    for n in sc["notes"]:
        lines.append(f"NOTE {n}")
    for label, res in sc["runs"].items():
        lines.append(f"== {label}: lock {res.get('lock')}  {res.get('role', '')}")
        for k in ("calls", "scored", "off_lock_dropped", "v8_refused_share_of_moe_time", "intruder_calls"):
            if k in res:
                lines.append(f"   {k}: {res[k]}")
        for mode, v in (res.get("span_over_kernel_sum_by_cg_mode") or {}).items():
            lines.append(f"   span / kernel-sum, cudagraph {mode:<8s} n={v['n']:<7d} median {v['median']:.2f}")
        for tile, v in (res.get("fills") or {}).items():
            sh, sv = v.get("share_of_in_serving_kernel_sum"), v.get("share_of_v8")
            lines.append(f"   API fills (replay) {tile:<22s} n={v['n']:<5d} {v['fills_us_median']:7.2f} us/call"
                         + (f"  = {100 * sh:5.2f}% of in-serving kernel-sum" if sh is not None else "")
                         + (f", {100 * sv:5.2f}% of v8" if sv is not None else ""))
        for tname, t in res["tables"].items():
            a = t["all"]
            if not a.get("n"):
                continue
            ci = a.get("MAPE_ci95")
            tag = " [primary]" if tname.endswith("vs kernel_sum_ms") else (
                " [secondary]" if tname.endswith("vs span_ms") else "")
            tname = tname + tag
            lines.append(f"   {tname:<52s} n={a['n']:<7d} MAPE {a['MAPE']:6.2f}%"
                         + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                         + f"  wMAPE {a['wMAPE']:6.2f}%  p50 {a['p50_abs_e']:5.2f}%  p95 {a['p95_abs_e']:6.2f}%"
                         f"  bias {a['bias']:+6.2f}%")
            for tile, m in t["tile"].items():
                lines.append(f"      {tile:<22s} n={m['n']:<7d} MAPE {m['MAPE']:6.2f}%  wMAPE {m['wMAPE']:6.2f}%"
                             f"  bias {m['bias']:+6.2f}%")
            pst = t.get("per_step_total", {})
            if pst.get("n"):
                lines.append(f"      per-step total        n={pst['n']:<7d} MAPE {pst['MAPE']:6.2f}%  "
                             f"bias {pst['bias']:+6.2f}%")
    for k, d in sc["deltas"].items():
        lines.append(f"== delta {k}: {d['what']}")
        for tile, v in d["per_tile_median_kernel_sum"].items():
            lines.append(f"      {tile:<22s} {v['a_ms'] * 1e3:9.2f} us vs {v['b_ms'] * 1e3:9.2f} us  {v['delta_pct']:+6.2f}%")
    return "\n".join(lines) + "\n"


def _kv(items) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not v:
            raise SystemExit(f"expected LABEL=PATH, got {it!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    ap.add_argument("--run", action="append", default=[], help="LABEL=DIR (nsys_extract'd serve_capture dir)")
    ap.add_argument("--replay", action="append", default=[], help="BRANCH=DIR (replay.py's run dir)")
    ap.add_argument("--sample-run", action="append", default=[], help="BRANCH=RUNLABEL the sample was drawn from")
    ap.add_argument("--replay-check", action="append", default=[],
                    help="BRANCH=replay_check.json (nsys_extract replay-check: each cell's API fills); absent: skipped")
    ap.add_argument("--frontier", default=None, help="frontier_op.py predict's directory (or a predictions.jsonl)")
    ap.add_argument("--sag", required=True, help="the board's 128x128 par (main.sh's sag1410 step: RESULTS/sag/par.json)")
    ap.add_argument("--pricer", default=None, help=f"pricer module (default {PL.PRICER_MODULE})")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = PL.MODELS[a.model]
    try:
        runs = {lab: load_run(lab, d, model, want_ids=True) for lab, d in _kv(a.run).items()}
        replay = {b: load_replay(d, b) for b, d in _kv(a.replay).items()}
        sample_run = {"decode": "A1710", "prefill": "A1410", **_kv(a.sample_run)}
        frontier = load_frontier(a.frontier) if a.frontier else {}
        sag128 = PL.load_board_par(a.sag)
        roof = PL.roofline_constants()
        prices = attach_predictions(model, runs, frontier, roof, sag128=sag128, module=a.pricer, workers=a.workers)
        fills = {b: load_replay_check(p) for b, p in _kv(a.replay_check).items()}
        sc = score(model=model, runs=runs, replay=replay, sample_run=sample_run, frontier=frontier, roof=roof,
                   prices=prices, sag128=sag128, fills=fills)
    except (PL.KitError, PL.RingError, OSError, ValueError, KeyError) as exc:
        print(f"FAIL score_prod: {type(exc).__name__}: {exc}")
        return 3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prod.score.json").write_text(json.dumps(sc, indent=1, default=lambda o: None if isinstance(o, float)
                                                    and math.isnan(o) else str(o)))
    txt = render(sc)
    (out / "prod.score.txt").write_text(txt)
    print(txt)
    if sc["pricing"]["errors"]:
        print(f"FAIL pricing: {sc['pricing']['errors']} priced calls raised (not refusals): "
              f"{sc['pricing']['first_errors']}")
        return 3
    print("PASS score_prod (descriptive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
