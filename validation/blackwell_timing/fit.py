#!/usr/bin/env python3
"""Turn a results directory of the timing microbenchmarks into model parameters.

    python fit.py RESULTS_DIR [--predicted DIR] [--config YAML] [--out DIR]
                  [--stat min|median] [--time event_ns|wall_ns]

RESULTS_DIR holds one `<bench>.out` per benchmark, as run_all.sh writes them (the
programs' stdout: one DEVICE line, one BENCH line per kernel launch).

Writes into --out (default RESULTS_DIR/fit):
  measured.csv      (a) per configuration: ns per repetition (min and median over trials),
                        trial-to-trial spread, %clock64 ticks per repetition
  derived.yaml      (b) the timing-model parameters these measurements imply, in the schema
                        of warplens/configs/*.yaml (a fragment: only what was measured)
  calibrated.yaml       the config given by --config with those entries replaced; loads with
                        warplens.timing.config.GpuConfig.load(path)
  comparison.csv    (c) per configuration: measured, the simulator's prediction with the
                        current config (from --predicted, written by predict.py), the ratio,
                        and how many SMs the prediction assumed the launch keeps busy
  report.md             all of the above as text; also printed

Everything is a slope. A configuration is launched with two repetition counts; "ns per
repetition" is the slope of launch time against repetitions, which removes launch
overhead and kernel setup. Parameters come from those slopes as described in README.md
and next to each formula below.

Needs only the standard library and PyYAML. It does not import warplens.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent.parent / "warplens" / "configs" / "b200.yaml"
DEFAULT_PREDICTED = HERE / "predicted"
NUMERIC = ("reps", "trial", "event_ns", "wall_ns", "clk", "check", "norm", "bytes", "m", "n", "k",
           "batch", "nctas", "sim_ns", "sim_instr", "sim_active_sms", "sim_ctas_per_sm", "sim_resident_ctas")


# -- reading ------------------------------------------------------------------------------
def parse_line(line: str) -> dict:
    out = {}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        out[key] = value.strip('"')
    return out


def read_dir(path) -> tuple[dict, list[dict], list[str]]:
    """(device, rows, problems) from every *.out file in `path`."""
    device, rows, problems = {}, [], []
    files = sorted(Path(path).glob("*.out"))
    if not files:
        raise SystemExit(f"{path}: no *.out files")
    for file in files:
        seen = False
        for line in file.read_text(errors="replace").splitlines():
            if line.startswith("DEVICE "):
                info = parse_line(line)
                device.setdefault("benches", []).append(info.get("bench", file.stem))
                for key, value in info.items():
                    if key not in ("bench",):
                        device.setdefault(key, value)
                if info.get("synthetic", "0") != "0":
                    device["synthetic"] = info["synthetic"]
            elif line.startswith("BENCH_ERROR "):
                problems.append(f"{file.name}: {line[len('BENCH_ERROR '):]}")
            elif line.startswith("BENCH "):
                row = parse_line(line)
                for key in NUMERIC:
                    if key in row:
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            del row[key]
                row["file"] = file.stem
                rows.append(row)
                seen = True
        if not seen:
            problems.append(f"{file.name}: no BENCH lines (the benchmark did not run or failed at once)")
    return device, rows, problems


# -- statistics ---------------------------------------------------------------------------
def linfit(xs, ys) -> tuple[float, float]:
    """Least-squares (slope, intercept). Two points give the line through them."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx == 0:
        return float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def summarize(rows: list[dict], key: str = "event_ns") -> dict[str, dict]:
    """Per configuration id: ns per repetition and its spread. Warm-up launches (trial < 0)
    are ignored. Ids whose repetition count never varies (launch) get min/median times."""
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("trial", 0) >= 0 and key in row:
            by_id.setdefault(row["id"], []).append(row)
    out = {}
    for ident, group in by_id.items():
        norm = group[0].get("norm", 1.0) or 1.0
        reps = sorted({r["reps"] for r in group})
        info = {"id": ident, "bench": group[0]["file"], "launches": len(group), "norm": norm,
                "row": group[0]}
        times = [r[key] for r in group]
        info["launch_min_ns"], info["launch_median_ns"] = min(times), statistics.median(times)
        # The kernel's own count, for the configurations whose launch time is not a number of
        # round trips (clc_latency/succeed/*): `clk` is the %clock64 ticks the cluster at (0, 0)
        # spent requesting and `check` how many of its requests succeeded, so it made check + 1.
        own = [r["clk"] / (r["check"] + 1) for r in group if r.get("clk", 0) > 0 and r.get("check", -1) >= 0]
        if own:
            info["clk_per_request"], info["clk_per_request_median"] = min(own), statistics.median(own)
        if len(reps) >= 2:
            at = {x: [r[key] for r in group if r["reps"] == x] for x in reps}
            slope, intercept = linfit(reps, [min(at[x]) for x in reps])
            info["ns_per_rep"] = slope / norm
            info["intercept_ns"] = intercept
            trials = sorted({r["trial"] for r in group})
            per_trial = []
            for t in trials:
                mine = [r for r in group if r["trial"] == t]
                if len({r["reps"] for r in mine}) >= 2:
                    per_trial.append(linfit([r["reps"] for r in mine], [r[key] for r in mine])[0] / norm)
            info["trials"] = len(per_trial)
            if per_trial:
                median = statistics.median(per_trial)
                info["ns_per_rep_median"] = median
                info["ns_per_rep_trial_min"], info["ns_per_rep_trial_max"] = min(per_trial), max(per_trial)
                info["spread_pct"] = 100 * (max(per_trial) - min(per_trial)) / median if median else float("nan")
            for extra in ("clk", "sim_instr"):
                if all(extra in r for r in group):
                    values = {x: [r[extra] for r in group if r["reps"] == x] for x in reps}
                    info[f"{extra}_per_rep"] = linfit(reps, [min(values[x]) for x in reps])[0] / norm
        out[ident] = info
    return out


def value(table: dict, ident: str, stat: str = "min"):
    info = table.get(ident)
    if not info or "ns_per_rep" not in info:
        return None
    if stat == "median" and "ns_per_rep_median" in info:
        return info["ns_per_rep_median"]
    return info["ns_per_rep"]


def series(table: dict, template: str, xs, stat: str = "min"):
    """[(x, ns per rep)] for the ids `template.format(x)` that were measured."""
    points = [(x, value(table, template.format(x), stat)) for x in xs]
    return [(x, y) for x, y in points if y is not None]


# -- parameters ---------------------------------------------------------------------------
TILE_BYTES = (64, 256, 1024, 4096, 16384, 32768)
CONFLICTS = (1, 2, 4, 8, 16, 32)
BATCHES = (1, 2, 4, 8)
MMA_K = {"f16": 16, "tf32": 8, "i8": 32}
# Cluster launch control. Requests that fail, in a launch of one cluster: the slope of launch
# time against requests is one round trip. In order of preference for clc_latency_ns: the
# first has nothing in its loop but the request; the other two add a cluster barrier.
CLC_FAIL = ("clc_latency/fail/plain/ctas=1", "clc_latency/fail/plain/ctas=2", "clc_latency/fail/multicast/ctas=2")
# Requests that succeed: on a GPU many clusters start at once, so only the kernel's own
# count (clk_per_request) is a round trip, and it is in %clock64 ticks.
CLC_SUCCEED = ("clc_latency/succeed/plain/ctas=1", "clc_latency/succeed/multicast/ctas=2")
CLC_DISAGREE = 0.2  # another form's latency further than this from the fitted one is named in its note
UNITS = {"sm_count": "SMs", "clock_ghz": "GHz", "hbm_capacity_gb": "GB", "hbm_bandwidth_tb_s": "TB/s",
         "l2_bandwidth_per_sm_gb_s": "GB/s", "tma_latency_ns": "ns", "clc_latency_ns": "ns",
         "tma_bytes_per_clk": "bytes/clk",
         "smem_wavefront_clk": "clk", "global_access_latency_clk": "clk", "tmem_access_clk": "clk",
         "tensor_core_tflops_f16": "TFLOPS", "tensor_core_tflops_tf32": "TFLOPS", "mma_latency_clk": "clk"}


def derive(meas: dict, pred: dict, cfg: dict, device: dict, stat: str = "min"):
    """(parameters, diagnostics). `cfg` is {key: value} of the config the predictions were
    made with. Each parameter: dict(key, value, formula, note). Parameters whose benchmark
    is missing are left out."""
    params, diag = [], []
    m = lambda ident: value(meas, ident, stat)
    p = lambda ident: value(pred, ident)

    def add(key, val, formula, note=""):
        if val is None or not math.isfinite(val):
            diag.append((key, "not derived", "", "the formula did not give a finite value: " + formula))
            return
        if key != "sm_count" and val < 0:
            note = (note + " " if note else "") + f"The formula gave {val:.4g}; clamped to 0, the model's lower limit."
            val = 0.0
        if val == 0 and key in ("clock_ghz", "tma_bytes_per_clk", "smem_wavefront_clk",
                                "l2_bandwidth_per_sm_gb_s", "hbm_bandwidth_tb_s",
                                "tensor_core_tflops_f16", "tensor_core_tflops_tf32"):
            diag.append((key, "not derived", "", "the measurement gave a non-positive value: " + formula))
            return
        params.append({"key": key, "value": val, "formula": formula, "note": note})

    # Device facts.
    if device.get("sm_count"):
        add("sm_count", float(device["sm_count"]), "cudaGetDeviceProperties().multiProcessorCount")
    if device.get("total_mem_bytes"):
        add("hbm_capacity_gb", round(float(device["total_mem_bytes"]) / 2 ** 30, 1),
            "cudaMemGetInfo() total bytes / 2^30",
            "The driver reports usable memory, which is below the nominal size.")
    sm_count = float(device.get("sm_count") or cfg["sm_count"])

    # clock_ghz: the model issues one PTX instruction per clock per warp.
    f_cfg = cfg["clock_ghz"]
    f = f_cfg
    a, b = m("alu_chain/warps=1"), p("alu_chain/warps=1")
    if a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
        f = f_cfg * b / a
        instr = pred["alu_chain/warps=1"].get("sim_instr_per_rep")
        add("clock_ghz", f,
            f"clock_ghz(config) {f_cfg:g} * predicted {b:.4g} / measured {a:.4g} ns per repetition of alu_chain/warps=1"
            + (f" ({instr:.3g} PTX instructions per repetition)" if instr else ""),
            "Effective issue clock for one warp of register-only instructions, not the core clock.")
    elif a is not None:
        diag.append(("clock_ghz", "not derived", "", "alu_chain has no positive finite timing slope; "
                     "clock-based fits retain the config clock"))
    clock_note = (f"Clocks converted with the fitted clock_ghz = {f:.4g}." if f != f_cfg else
                  f"Clocks converted with the config's clock_ghz = {f_cfg:g} (no usable alu_chain clock fit).")
    ticks = meas.get("alu_chain/warps=1", {})
    ticks_per_ns = None  # of %clock64, to turn a kernel's own tick count into time
    if ticks.get("clk_per_rep") and a:
        diag.append(("%clock64 ticks per ns", ticks["clk_per_rep"] / a, "ticks/ns",
                     "alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock"))
        if a > 0 and math.isfinite(ticks["clk_per_rep"] / a) and ticks["clk_per_rep"] > 0:
            ticks_per_ns = ticks["clk_per_rep"] / a

    # smem_wavefront_clk: slope of ns per access against wavefronts per access.
    pts = series(meas, "smem_wavefronts/load_chain/conflict={}/warps=1", CONFLICTS, stat)
    if len(pts) >= 2:
        slope, intercept = linfit(*zip(*pts))
        add("smem_wavefront_clk", slope * f,
            f"slope {slope:.4g} ns per extra wavefront (load_chain, conflict {pts[0][0]}..{pts[-1][0]}) * clock_ghz",
            clock_note)
        diag.append(("shared load, no conflict", intercept + slope, "ns", "smem_wavefronts/load_chain fit at conflict=1"))

    # Fixed costs: the value that makes the model reproduce the measured loop.
    def fixed(key, ident, what):
        a, b = m(ident), p(ident)
        if a is None or b is None:
            return
        add(key, cfg[key] + a * f - b * f_cfg,
            f"config {cfg[key]:g} + measured {a:.4g} ns * clock_ghz - predicted {b:.4g} ns * {f_cfg:g} ({ident}, {what})",
            clock_note)

    fixed("global_access_latency_clk", "global_load/load_chain/footprint=l2", "one load per repetition")
    fixed("tmem_access_clk", "tmem_access/ld/x1", "one tcgen05.ld per repetition")

    # clc_latency_ns. The model lands the response of a clusterlaunchcontrol.try_cancel a fixed
    # time after the request is issued, the same for every form (docs/clc.md, timing/model.py
    # kind "clc"). A request's round trip (arm the mbarrier, try_cancel, wait for the phase) is
    # that latency plus the loop's instructions, which the simulator counted for the same loop:
    #     latency = measured round trip - (predicted round trip - config latency)
    # with the instruction time rescaled to the fitted clock, as for tma_latency_ns.
    clc_measured = [i for i in (*CLC_FAIL, *CLC_SUCCEED) if i in meas]
    clc_values = {}  # id -> (latency this form implies, measured round trip, instruction time)
    if clc_measured and "clc_latency_ns" not in cfg:
        diag.append(("clc_latency_ns", "not derived", "", "the config the predictions were made with has no "
                     "clc_latency_ns entry, so the instruction time of a request cannot be separated"))
        clc_measured = []
    for ident in clc_measured:
        label = f"CLC request, {ident[len('clc_latency/'):]}"
        b = p(ident)
        if ident in CLC_SUCCEED:
            own = meas[ident].get("clk_per_request_median" if stat == "median" else "clk_per_request")
            if own is None:
                diag.append((label, "not derived", "", "no %clock64 ticks in its lines (clk = 0): the launch time of a "
                             "grid of clusters is not a number of round trips, only the kernel's own count is"))
                continue
            if ticks_per_ns is None:
                diag.append((label, "not derived", "", f"{own:.4g} %clock64 ticks per request, but no ticks per ns "
                             "to convert them (alu_chain/warps=1 is missing or its clock did not run)"))
                continue
            a, how = own / ticks_per_ns, (f"{own:.4g} ticks per request of the first cluster (clk / (check + 1)) / "
                                          f"{ticks_per_ns:.4g} ticks per ns")
        else:
            a, how = m(ident), "slope of launch time against requests"
        if a is None or not math.isfinite(a) or a <= 0:
            diag.append((label, "not derived", "", "no positive finite round trip was measured"))
            continue
        if b is None or not math.isfinite(b):
            diag.append((label, a, "ns", how + "; round trip only: there is no prediction of this configuration "
                         "to take the instruction time from"))
            continue
        overhead = (b - cfg["clc_latency_ns"]) * f_cfg / f
        clc_values[ident] = (a - overhead, a, overhead)
        diag.append((label, a, "ns", f"{how}; minus {overhead:.4g} ns of instructions the simulator counts for the "
                     f"same loop = {a - overhead:.4g} ns of latency"))
    chosen = next((i for i in CLC_FAIL if i in clc_values), None)
    if chosen:
        latency, a, overhead = clc_values[chosen]
        note = ("One latency for every form of the request, as the model has. Includes whatever the mbarrier "
                "wake-up costs on hardware beyond its instructions. " + clock_note)
        if chosen != CLC_FAIL[0]:
            note += (f" {CLC_FAIL[0]} was not usable; this loop also has a cluster barrier per request, which "
                     "the model charges as instructions only.")
        others = {i: v[0] for i, v in clc_values.items()
                  if i != chosen and abs(v[0] - latency) > CLC_DISAGREE * max(abs(latency), 1.0)}
        if others:
            note += (" NOT ONE NUMBER on this device: " + ", ".join(f"{i} implies {v:.4g} ns" for i, v in others.items())
                     + "; the model would need a latency per form.")
        add("clc_latency_ns", latency,
            f"measured {a:.4g} ns per request - {overhead:.4g} ns of instruction time the simulator counts for "
            f"the same loop ({chosen}, requests that fail in a launch of one cluster)", note)
    elif clc_measured:
        diag.append(("clc_latency_ns", "not derived", "", "none of " + ", ".join(CLC_FAIL) + " has a usable "
                     "measurement and prediction; the requests that succeed are reported, not fitted, because "
                     "their round trip rests on the %clock64 tick rate"))

    # TMA: round trip against bytes.
    fits = {}
    for name in ("load/warm", "store", "load/cold"):
        template = "tma/" + name + "/bytes={}/ctas=1"
        pts, ppts = series(meas, template, TILE_BYTES, stat), series(pred, template, TILE_BYTES)
        if len(pts) >= 2:
            fits[name] = linfit(*zip(*pts)) + (linfit(*zip(*ppts)) if len(ppts) >= 2 else (None, None))
            slope = fits[name][0]
            diag.append((f"TMA {name}: ns per byte", slope, "ns/byte",
                         f"= {1 / slope:.4g} GB/s for one SM" if slope > 0 else "non-positive slope"))
            diag.append((f"TMA {name}: intercept", fits[name][1], "ns", "round trip of a zero-byte copy, extrapolated"))
    rates = {name: 1 / fit[0] for name, fit in fits.items() if name != "load/cold" and fit[0] > 0}
    if rates:
        engine = max(rates.values())
        which = max(rates, key=rates.get)
        add("tma_bytes_per_clk", engine / f,
            f"{engine:.4g} bytes/ns (1 / slope of round trip against bytes, {which}) / clock_ghz",
            "The faster of warm loads and stores is taken as the engine rate: the model charges "
            "max(engine, memory) per copy, so each slope is the slower of the two. " + clock_note)
        if "load/warm" in rates:
            bound = rates["load/warm"] >= 0.9 * engine
            add("l2_bandwidth_per_sm_gb_s", rates["load/warm"],
                "1 / slope of round trip against bytes, warm loads, in bytes/ns",
                "LOWER BOUND: warm loads run at the engine rate, so L2 is at least this fast; the "
                "model reproduces the measurement with any value from here up." if bound else
                "Warm loads are slower than stores, so this is the rate of the L2-to-SMEM path.")
    if "load/warm" in fits and fits["load/warm"][3] is not None:
        _, intercept, _, predicted = fits["load/warm"]
        overhead = (predicted - cfg["tma_latency_ns"]) * f_cfg / f
        add("tma_latency_ns", intercept - overhead,
            f"intercept {intercept:.4g} ns of warm-load round trip against bytes - {overhead:.4g} ns of "
            "instruction time the simulator counts for the same loop",
            "Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. " + clock_note)

    # HBM bandwidth. A cold copy's round trip is a fixed part (the intercept of the one-CTA
    # cold fit: latency and instructions) plus the transfer; only the transfer is bandwidth.
    stream = m("gstream/triad")
    fixed_ns = fits["load/cold"][1] if "load/cold" in fits else 0.0
    how = (f"CTAs * 16384 bytes / (round trip - {fixed_ns:.4g} ns fixed cost per copy, from the one-CTA cold fit)"
           if "load/cold" in fits else "CTAs * 16384 bytes / round trip (no one-CTA cold fit to remove the fixed cost)")
    cold_rate = {}
    for ctas in ("8", "37", "allsm"):
        info = meas.get(f"tma/load/cold/bytes=16384/ctas={ctas}")
        t = value(meas, info["id"], stat) if info else None
        if t and t > fixed_ns:
            cold_rate[ctas] = info["row"].get("nctas", sm_count) * 16384 / (t - fixed_ns)
            diag.append((f"cold TMA fetch rate, ctas={ctas}", cold_rate[ctas], "GB/s",
                         "total over the active SMs: " + how + ". The model gives each SM hbm_bandwidth / active_sms"))
    cold_tb = cold_rate["allsm"] / 1000 if "allsm" in cold_rate else None
    if stream and stream > 0:
        add("hbm_bandwidth_tb_s", 1 / stream / 1000,
            f"STREAM Triad: 1 / ({stream:.4g} ns per byte) / 1000 (gstream/triad, 24 bytes per element)",
            "Same definition as the config's published figure." +
            (f" Cold TMA loads on every SM gave {cold_tb:.4g} TB/s." if cold_tb else ""))
    elif cold_tb:
        add("hbm_bandwidth_tb_s", cold_tb, "cold TMA loads, one CTA per SM: " + how + " / 1000")

    # Tensor Core.
    for kind in ("f16", "tf32", "i8"):
        for ctas in ("allsm", "1"):
            template = f"mma_rate/kind={kind}/m=128/n=256/batch={{}}/ctas={ctas}"
            pts, ppts = series(meas, template, BATCHES, stat), series(pred, template, BATCHES)
            if len(pts) < 2:
                continue
            slope, intercept = linfit(*zip(*pts))
            tflops = 2 * 128 * 256 * MMA_K[kind] / slope * sm_count / 1000 if slope > 0 else float("nan")
            diag.append((f"MMA {kind} m128 n256, ctas={ctas}", slope, "ns/MMA",
                         f"= {tflops:.5g} TFLOPS for {sm_count:.0f} SMs; batch intercept {intercept:.4g} ns"))
            key = f"tensor_core_tflops_{kind}"
            if key in cfg and not any(q["key"] == key for q in params):
                add(key, tflops,
                    f"2 * 128 * 256 * {MMA_K[kind]} FLOP / {slope:.4g} ns per MMA * {sm_count:.0f} SMs / 1000 "
                    f"(slope over batch size, ctas={ctas})",
                    "" if ctas == "allsm" else "From ONE active SM, scaled by the SM count; the all-SM variant was not measured.")
            if kind == "f16" and ctas == "1" and len(ppts) >= 2:
                predicted = linfit(*zip(*ppts))[1]
                add("mma_latency_clk", cfg["mma_latency_clk"] + intercept * f - predicted * f_cfg,
                    f"config {cfg['mma_latency_clk']:g} + measured batch intercept {intercept:.4g} ns * clock_ghz - "
                    f"predicted {predicted:.4g} ns * {f_cfg:g} (f16 m128 n256)",
                    "Includes whatever tcgen05.commit and the mbarrier wake-up cost on hardware beyond "
                    "their instructions. " + clock_note)

    # Diagnostics with no parameter.
    for ident, label in (("mbar/arrive_wait/threads=1", "mbarrier arrive + wait, one thread"),
                         ("mbar/arrive_wait/threads=32", "mbarrier arrive + wait, whole warp")):
        if m(ident) is not None:
            diag.append((label, m(ident), "ns", f"{ident}; predicted {p(ident):.4g} ns" if p(ident) else ident))
    for op in ("ld", "st"):
        a, b = m(f"tmem_access/{op}/x1"), m(f"tmem_access/{op}/x1/nowait")
        if a is not None and b is not None:
            diag.append((f"tcgen05.wait::{op}", a - b, "ns", f"tmem_access/{op}/x1 minus the same loop without the wait; the model charges one clock"))
    return params, diag


# -- output -------------------------------------------------------------------------------
def provenance(device: dict, results: str) -> tuple[str, list[str]]:
    """(source text for the YAML entries, warning lines)."""
    name = device.get("name", "unknown device")
    cc = device.get("cc", "?")
    if device.get("synthetic", "0") != "0":
        return ("assumption: SYNTHETIC test data, not a measurement",
                ["SYNTHETIC input: these numbers were generated, not measured. Do not use them."])
    if not cc.startswith("10."):
        return (f"assumption: measured on {name} (compute capability {cc}), which is not a B200",
                [f"Measured on {name} (compute capability {cc}), NOT a Blackwell data-center GPU. "
                 "These are not B200 parameters."])
    return f"measured: validation/blackwell_timing on {name}, results {results}", []


def yaml_fragment(params, device, results, warnings) -> str:
    source, _ = provenance(device, results)
    lines = ["# Timing-model parameters derived by validation/blackwell_timing/fit.py.",
             f"# Results: {results}. Device: {device.get('name', '?')} (compute capability {device.get('cc', '?')}).",
             "# A fragment in the schema of warplens/configs/*.yaml: entries not listed were not measured."]
    lines += [f"# WARNING: {w}" for w in warnings]
    for q in params:
        val = int(q["value"]) if q["key"] == "sm_count" else float(f"{q['value']:.5g}")
        note = (q["formula"] + ". " + q["note"]).strip().replace('"', "'")
        lines += ["", f"{q['key']}:", f"  value: {val}", f"  unit: {UNITS[q['key']]}",
                  f'  source: "{source}"', f'  note: "{note}"']
    return "\n".join(lines) + "\n"


def merged_config(base: Path, fragment: str, warnings) -> str:
    import yaml
    config = yaml.safe_load(base.read_text())
    for key, entry in (yaml.safe_load(fragment) or {}).items():
        config[key] = entry
    head = [f"# {base.name} with the entries measured by validation/blackwell_timing replaced (fit.py).",
            "# Entries whose source does not mention validation/blackwell_timing are unchanged."]
    head += [f"# WARNING: {w}" for w in warnings]
    return "\n".join(head) + "\n" + yaml.safe_dump(config, sort_keys=False, width=1000)


def fmt(x, digits=4) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.{digits}g}"


def placement_problems(device: dict, placement: dict, meas: dict, pred: dict) -> list[str]:
    """Where the run contradicts the active-SM assumption its predictions were made with
    (predict.py: meta.json `placement`, and sim_active_sms on every predicted launch)."""
    out = []
    if not placement:
        if pred and any("sim_active_sms" not in info["row"] for info in pred.values()):
            out.append("the predictions record no active-SM assumption (an older predict.py made them): every "
                       "launch was timed as if all SMs were busy")
        return out
    sms = device.get("sm_count")
    try:
        sms = float(sms) if sms else None
    except ValueError:
        sms = None
    if sms and placement.get("sm_count") and sms != placement["sm_count"]:
        out.append(f"the device has {sms:.0f} SMs and the predictions place launches on {placement['sm_count']}: "
                   "every configuration that fills the device was predicted for another number of active SMs")
    for key, assumed in (placement.get("limits") or {}).items():
        try:
            reported = float(device.get(key) or 0)
        except ValueError:
            continue
        if reported and reported != assumed:
            out.append(f"the device reports {key} = {reported:.0f}; the predictions' active-SM assumption used "
                       f"{assumed}: CTAs per SM, and with them the CTAs resident at once, were computed for other limits")
    if sms and placement.get("mode") == "launch":
        differ = []
        for ident, info in meas.items():
            row, predicted = info["row"], pred.get(ident, {}).get("row", {})
            # `nctas` is the grid of the launch, except in clc_latency/succeed/*, where it is the cluster size.
            if ident in CLC_SUCCEED or "nctas" not in row or "sim_active_sms" not in predicted:
                continue
            implied = min(row["nctas"], sms)
            if implied != predicted["sim_active_sms"]:
                differ.append(f"`{ident}` ({row['nctas']:.0f} CTAs launched, {implied:.0f} SMs; predicted for "
                              f"{predicted['sim_active_sms']:.0f})")
        if differ:
            out.append(f"{len(differ)} configuration(s) were launched with a grid that keeps another number of SMs "
                       "busy than their prediction assumed, e.g. " + ", ".join(differ[:3]))
    return out


def build_report(device, meas, pred, params, diag, cfg, problems, warnings, pred_status) -> str:
    out = ["# Timing calibration report", ""]
    out += [f"**WARNING: {w}**" for w in warnings] + ([""] if warnings else [])
    out += [f"Device: {device.get('name', '?')}, compute capability {device.get('cc', '?')}, "
            f"{device.get('sm_count', '?')} SMs, {device.get('total_mem_bytes', '?')} bytes."]
    if problems:
        out += ["", "Problems in the results:"] + [f"- {x}" for x in problems]
    out += ["", "## (a) Measured", "",
            "ns per repetition is the slope of launch time against repetitions. `min` uses the fastest "
            "launch at each repetition count; `median` is the median of the per-trial slopes; spread is "
            "(max - min) / median of the per-trial slopes.", "",
            "| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |", "|---|---|---|---|---|---|"]
    for ident, info in meas.items():
        if "ns_per_rep" in info:
            out.append(f"| {ident} | {fmt(info['ns_per_rep'])} | {fmt(info.get('ns_per_rep_median'))} | "
                       f"{fmt(info.get('spread_pct'), 3)} | {info.get('trials', 0)} | {fmt(info.get('clk_per_rep'))} |")
        else:
            out.append(f"| {ident} (whole launch) | {fmt(info['launch_min_ns'])} | {fmt(info['launch_median_ns'])} | | "
                       f"{info['launches']} | |")
    out += ["", "## (b) Derived model parameters", "",
            "| parameter | derived | config now | derived / config | how |", "|---|---|---|---|---|"]
    for q in params:
        now = cfg.get(q["key"])
        out.append(f"| `{q['key']}` | {fmt(q['value'], 5)} {UNITS[q['key']]} | {fmt(now, 5)} | "
                   f"{fmt(q['value'] / now) if now else 'n/a'} | {q['formula']}. {q['note']} |")
    missing = [k for k in UNITS if k not in {q["key"] for q in params}]
    if missing:
        out += ["", "Not derived (benchmark missing from the results, or unusable): " + ", ".join(f"`{k}`" for k in missing)]
    out += ["", "Other measured quantities:", "", "| quantity | value | unit | note |", "|---|---|---|---|"]
    out += [f"| {name} | {fmt(val, 5) if not isinstance(val, str) else val} | {unit} | {note} |" for name, val, unit, note in diag]
    out += ["", "## (c) Measured against the simulator's current prediction", "",
            "Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition. "
            "\"SMs assumed\" is how many SMs the prediction took the launch to keep busy (they share HBM in "
            "the model); it comes from the launch's shape, and nothing in a run measures it directly. What a "
            "run can contradict is listed under \"Problems in the results\".", "",
            "| configuration | measured | predicted | measured / predicted | SMs assumed |", "|---|---|---|---|---|"]
    for ident, info in meas.items():
        pi = pred.get(ident)
        if "ns_per_rep" in info:
            a, b = info["ns_per_rep"], pi.get("ns_per_rep") if pi else None
            unit = ""
        else:
            a, b = info["launch_min_ns"], pi.get("launch_min_ns") if pi else None
            unit = " (launch)"
        if b is None:
            ratio = "no prediction"
        elif b <= 1e-9 * max(abs(a), 1e-12) or b <= 0:
            ratio = "n/a: the model predicts no cost"
        else:
            ratio = fmt(a / b)
        assumed = (pi or {}).get("row", {}).get("sim_active_sms")
        out.append(f"| {ident}{unit} | {fmt(a)} | {fmt(b)} | {ratio} | {'' if assumed is None else f'{assumed:.0f}'} |")
    if any(i in meas for i in CLC_SUCCEED):
        out += ["", "`clc_latency/succeed/*`: on a GPU many clusters start at once, so these two slopes are not "
                "round trips and their ratios mean nothing; the kernel's own count is under \"Other measured "
                "quantities\" (CLC request, succeed/...)."]
    absent = [i for i in pred if i not in meas]
    if absent:
        out += ["", f"{len(absent)} predicted configuration(s) have no measurement, e.g. " +
                ", ".join(f"`{i}`" for i in absent[:5])]
    bad = {b: s for b, s in pred_status.items() if s.get("status") != "ok"}
    if bad:
        out += ["", "Benchmarks the simulator could not run:"] + [f"- `{b}`: {s.get('reason', '')}" for b, s in bad.items()]
    return "\n".join(out) + "\n"


def load_config_values(path: Path) -> dict:
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    return {k: float(v["value"]) for k, v in raw.items() if isinstance(v, dict) and "value" in v}


def load_placement(path: Path) -> dict:
    """The active-SM assumption recorded with the predictions in `path`; {} if there is none."""
    import json
    meta = Path(path) / "meta.json"
    return (json.loads(meta.read_text()).get("placement") or {}) if meta.exists() else {}


def load_predicted(path: Path) -> tuple[dict, dict, dict]:
    """(per-id summary of the simulator's times, config values used, per-benchmark status)."""
    import json
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    _, rows, _ = read_dir(path)
    return summarize(rows, "sim_ns"), meta.get("config", {}), meta.get("benchmarks", {})


def run(results: Path, predicted: Path = DEFAULT_PREDICTED, config: Path = DEFAULT_CONFIG,
        out: Path | None = None, stat: str = "min", time_key: str = "event_ns") -> dict:
    results = Path(results)
    device, rows, problems = read_dir(results)
    meas = summarize(rows, time_key)
    pred, pred_cfg, pred_status = load_predicted(predicted)
    cfg = load_config_values(config)
    if pred_cfg and any(abs(pred_cfg.get(k, v) - v) > 1e-12 * abs(v) for k, v in cfg.items()):
        problems.append(f"the predictions in {predicted} were made with a config that differs from {config}; "
                        "the derived parameters use the predictions' config")
    problems += placement_problems(device, load_placement(predicted), meas, pred)
    cfg_used = {**cfg, **pred_cfg}
    params, diag = derive(meas, pred, cfg_used, device, stat)
    _, warnings = provenance(device, results.name)
    fragment = yaml_fragment(params, device, results.name, warnings)
    report = build_report(device, meas, pred, params, diag, cfg_used, problems, warnings, pred_status)
    out = Path(out) if out else results / "fit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "derived.yaml").write_text(fragment)
    (out / "calibrated.yaml").write_text(merged_config(Path(config), fragment, warnings))
    (out / "report.md").write_text(report)
    with (out / "measured.csv").open("w", newline="") as handle:
        fields = ["id", "bench", "ns_per_rep", "ns_per_rep_median", "ns_per_rep_trial_min", "ns_per_rep_trial_max",
                  "spread_pct", "trials", "intercept_ns", "clk_per_rep", "launch_min_ns", "launch_median_ns", "launches"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(meas.values())
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "measured_ns_per_rep", "predicted_ns_per_rep", "measured_over_predicted",
                         "predicted_active_sms"])
        for ident, info in meas.items():
            a = info.get("ns_per_rep", info["launch_min_ns"])
            pi = pred.get(ident, {})
            b = pi.get("ns_per_rep", pi.get("launch_min_ns")) if "ns_per_rep" in info else pi.get("launch_min_ns")
            assumed = pi.get("row", {}).get("sim_active_sms")
            writer.writerow([ident, a, "" if b is None else b, a / b if b and b > 0 else "",
                             "" if assumed is None else int(assumed)])
    return {"device": device, "measured": meas, "predicted": pred, "params": {q["key"]: q for q in params},
            "diagnostics": diag, "report": report, "out": out, "problems": problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--predicted", type=Path, default=DEFAULT_PREDICTED)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stat", choices=["min", "median"], default="min")
    parser.add_argument("--time", choices=["event_ns", "wall_ns"], default="event_ns")
    args = parser.parse_args(argv)
    result = run(args.results, args.predicted, args.config, args.out, args.stat, args.time)
    print(result["report"])
    print(f"written to {result['out']}/: report.md measured.csv derived.yaml calibrated.yaml comparison.csv",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
