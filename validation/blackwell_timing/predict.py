#!/usr/bin/env python3
"""Run every timing microbenchmark in the simulator and record what the model predicts.

    PYTHONPATH=<repo> python predict.py [--out predicted] [--config YAML] [--bench NAME ...]

Each program in bin/sim (build.sh sim) is run as `warplens run --timing --model B200
bin/sim/<bench> --sim`. The command is called in-process only so that the launch records
can be read with full precision; the summary it prints rounds to whole nanoseconds.

Writes into --out:
  <bench>.out   the program's DEVICE and BENCH lines, each BENCH line with the simulator's
                numbers for that launch appended: sim_ns (kernel time), sim_instr
                (warp-instructions), sim_bound (the model's bottleneck verdict), and the
                placement the launch was timed with: sim_active_sms (SMs assumed busy, which
                share HBM), sim_ctas_per_sm (occupancy limit), sim_resident_ctas
  meta.json     the config values used, and per benchmark whether the simulator ran it
                (if not, why) and, per configuration, the active-SM assumption with the
                launch shape it comes from. With --bench, the entries of the other benchmarks
                are kept from the meta.json already there, if it was made the same way
  predictions.md  a table of ns per repetition per configuration, with the SMs assumed active

Active SMs. The timing model gives each SM 1/active_sms of the HBM bandwidth. Each launch
is timed with the number its own shape implies (warplens/timing/placement.py: grid, cluster
size, CTAs per SM by shared memory and threads, the config's SM count), not with "every
SM". Where the hardware build launches a larger grid than the simulator build, the hardware
grid is used (HARDWARE_CLUSTERS). --placement all-sms restores the old assumption, to
compare the two against a measurement.

--model selects the model and its default config (B200 unless specified).
--config runs with another timing config in place of that default, e.g.
the calibrated.yaml that fit.py writes, to check that the calibrated model reproduces
the measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
sys.path.insert(0, str(HERE.parent / "blackwell"))
from probe_process import forward_interrupts, run_process  # noqa: E402

BENCHES = ["launch", "alu_chain", "smem_wavefronts", "global_load", "gstream", "cp_async",
           "mbar", "tma", "mma_rate", "tmem_access", "clc_latency", "cluster_costs", "pipelined_gemm",
           "ncu_banks"]
# What the simulator cannot run, by design of the benchmark. Recorded in meta.json.
NOT_SIMULATED = {
    "mma_rate/kind=i8": "warplens implements tcgen05.mma for kind::f16 and kind::tf32 only, and the "
                        "timing model has no rate for other kinds; the i8 variant is hardware-only",
    "%clock64": "warplens does not implement the %clock64 register; the simulator build (BENCH_CLOCK=0) "
                "omits the two clock reads around each loop, so clk is 0 in every predicted line",
}
# Kernels whose grid is smaller with --sim than on hardware: {part of the kernel's name: the
# fewest clusters a hardware launch of it has}. The active-SM assumption is made for the
# hardware grid, because that is the launch the prediction is compared with.
#   triad_kernel        gstream launches n / 256 CTAs, n = 2^27 and 2^28 elements when memory
#                       allows (2^19 CTAs at least on a B200); 8 and 16 CTAs with --sim
#   clc_succeed_kernel  clc_latency launches reps + 1 clusters, reps = 10000 and 20000;
#                       17 and 33 with --sim, where the first cluster takes over the rest
HARDWARE_CLUSTERS = {"triad_kernel": 1 << 19, "clc_succeed_kernel": 10001}
PLACEMENTS = ("launch", "all-sms")
PLACEMENT_RULE = {
    "launch": "per launch, from its shape: active SMs = min(SM count, CTAs resident at once), CTAs spread "
              "over the SMs before any SM takes a second one, a cluster's CTAs on different SMs; resident "
              "CTAs = min(grid, SM count x CTAs per SM); CTAs per SM = min(shared memory per SM / per CTA, "
              "threads per SM / per CTA, resident-CTA limit). See warplens/timing/placement.py",
    "all-sms": "every SM of the config is assumed active for every launch (--placement all-sms)",
}


def install_placement(placement: str = "launch") -> dict:
    """Make every timed launch of this process use the active SMs its own shape implies.

    `warplens run` has one --active-sms for a whole program, and the benchmarks launch several
    shapes each, so the number is set here, per launch: `warplens.host.device.run_kernel` is
    replaced by a wrapper that computes the placement (warplens/timing/placement.py, for the
    hardware grid where HARDWARE_CLUSTERS names one) and, with placement "launch", gives the
    launch a Timing with that many active SMs. With "all-sms" the launch keeps every SM and
    the placement is only recorded. Returns {id(LaunchStats): placement as a dict}, filled as
    launches finish."""
    from warplens.host import device as host_device
    from warplens.host.launch import dim3
    from warplens.timing.model import Timing
    from warplens.timing.placement import place
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    placed = {}
    run_kernel = host_device.run_kernel

    def run_placed(kernel, mem, *, grid, block, smem=0, timing=None, cluster=None, **rest):
        shape = None
        if timing is not None:
            try:
                grid_ctas = math.prod(dim3(grid, "grid"))
                cluster_ctas = math.prod(dim3(cluster, "cluster")) if cluster is not None else 1
                hardware = next((n * cluster_ctas for part, n in HARDWARE_CLUSTERS.items()
                                 if part in kernel.fn.name), grid_ctas)
                shape = place(max(grid_ctas, hardware), sm_count=int(timing.config["sm_count"]),
                              cluster_ctas=cluster_ctas, block_threads=math.prod(dim3(block, "block")),
                              smem_bytes=kernel.smem_bytes(int(smem)))
            except Exception:  # an invalid launch: run_kernel refuses it with its own message
                shape = None
            if shape is not None:
                if placement == "launch":
                    timing = Timing(timing.config, shape.active_sms)
                shape = {**shape.as_dict(), "sim_grid_ctas": grid_ctas, "derivation": shape.derivation}
        stats = run_kernel(kernel, mem, grid=grid, block=block, smem=smem, timing=timing, cluster=cluster, **rest)
        if shape is not None:
            placed[id(stats)] = shape
        return stats

    host_device.run_kernel = run_placed
    return placed


def one(program: str, record: str, config: str | None, model: str, placement: str = "launch") -> int:
    """Child process: `warplens run --timing program --sim`, launch records to `record`."""
    from warplens import cli
    if config:
        import shutil
        from warplens.timing import config as timing_config
        scratch = Path(tempfile.mkdtemp(prefix="warplens_timing_config_"))
        shutil.copy(config, scratch / f"{model.lower()}.yaml")
        timing_config.CONFIG_DIR = scratch
    launches = []
    placed = install_placement(placement)  # id(LaunchStats) -> what the launch was placed as

    def capture(done, _model):
        launches.extend(done)

    cli.summary = capture
    code = cli.main(["run", "--timing", "--model", model, program, "--sim"])
    Path(record).write_text(json.dumps([
        {"kernel": name, "ctas": stats.ctas, "instructions": stats.instructions,
         "time_ns": stats.timing.time_ns, "bound": stats.timing.bottleneck.replace(" ", "_"),
         "findings": len(stats.findings), "active_sms": stats.timing.active_sms,
         "placement": placed.get(id(stats))} for name, stats in launches]))
    return code


def placement_fields(sim: dict) -> str:
    """What is appended to a BENCH line about the placement of its launch. Nothing, if the
    launch record does not say how many SMs it was timed with."""
    if sim.get("active_sms") is None:
        return ""
    shape = sim.get("placement") or {}
    text = f" sim_active_sms={sim['active_sms']}"
    if shape:
        text += f" sim_ctas_per_sm={shape['ctas_per_sm']} sim_resident_ctas={shape['resident_ctas']}"
    return text


def assumption(sim: dict) -> dict:
    """The active-SM assumption of one launch, as meta.json records it per configuration.
    `active_sms` is what the launch was timed with; `spread_sms` what its shape implies (the
    same, unless --placement all-sms)."""
    shape = dict(sim.get("placement") or {})
    if shape:
        shape["spread_sms"] = shape.pop("active_sms")
        shape.pop("sm_count", None)
    return {"active_sms": sim.get("active_sms"), **shape}


def predict(out: Path, benches=None, config: Path | None = None, bin_dir: Path | None = None,
            model: str = "B200", timeout: int = 1800, quiet: bool = False, placement: str = "launch") -> dict:
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bin_dir = Path(bin_dir) if bin_dir else HERE / "bin" / "sim"
    config = Path(config) if config else fit.DEFAULT_CONFIG.with_name(model.lower() + ".yaml")
    status = {}
    for bench in benches or BENCHES:
        program = bin_dir / bench
        if not program.exists():
            status[bench] = {"status": "failed", "reason": f"{program} is not built (run build.sh sim)"}
            continue
        with tempfile.TemporaryDirectory() as scratch:
            record = Path(scratch) / "launches.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--one", str(program), "--record", str(record),
                       "--model", model, "--placement", placement]
            if config:
                command += ["--config", str(config)]
            process_status, _, output, errors = run_process(command, timeout, env=dict(
                os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join(
                    filter(None, [str(HERE.parent.parent), os.environ.get("PYTHONPATH")]))))
            code = 0 if process_status == "ok" else -1
            stdout, stderr = output.decode(errors="replace"), errors.decode(errors="replace")
            if process_status != "ok":
                stderr = process_status + "\n" + stderr
            launches = json.loads(record.read_text()) if record.exists() else []
        lines, at, assumed = [], 0, {}
        for line in stdout.splitlines():
            if line.startswith("BENCH "):
                if at < len(launches):
                    sim = launches[at]
                    tokens = [t if not t.startswith("wall_ns=") else "wall_ns=0" for t in line.split()]
                    line = " ".join(tokens) + (f" sim_ns={sim['time_ns']!r} sim_instr={sim['instructions']}"
                                               f" sim_bound={sim['bound']} sim_findings={sim['findings']}"
                                               + placement_fields(sim))
                    # One entry per configuration. Its launches differ in repetitions only; if
                    # one was placed on another number of SMs, the entry lists them all.
                    ident = fit.parse_line(line).get("id", "")
                    entry = assumed.setdefault(ident, assumption(sim))
                    if sim.get("active_sms") not in (None, entry["active_sms"]):
                        entry["active_sms_of_other_launches"] = sorted(
                            {*entry.get("active_sms_of_other_launches", []), sim["active_sms"]})
                at += 1
                lines.append(line)
            elif line.startswith(("DEVICE ", "BENCH_ERROR ")):
                lines.append(line)
        (out / f"{bench}.out").write_text("\n".join(lines) + "\n")
        if code == 0 and at == len(launches) and at:
            status[bench] = {"status": "ok", "launches": at, "active_sms": assumed}
        else:
            message = [l for l in stderr.splitlines() if l.strip()]
            status[bench] = {"status": "failed", "returncode": code, "launches": min(at, len(launches)),
                             "reason": " | ".join(message[:4]) or f"{at} BENCH lines for {len(launches)} launches",
                             "active_sms": assumed}
        if not quiet:
            print(f"{bench}: {status[bench]['status']}, {status[bench].get('launches', 0)} launch(es)"
                  + (f": {status[bench]['reason']}" if status[bench]["status"] != "ok" else ""), file=sys.stderr)
    values = fit.load_config_values(config or fit.DEFAULT_CONFIG)
    # Which nvcc built the programs the simulator ran (bin/sim/toolkit.txt, from build.sh). The
    # predicted instruction counts are those of its code; run_all.sh flags a GPU run built by another.
    toolkit = bin_dir / "toolkit.txt"
    nvcc_version = next((line.split("=", 1)[1] for line in (toolkit.read_text().splitlines() if toolkit.is_file() else [])
                         if line.startswith("nvcc_version=")), "unknown")
    # A run of some benchmarks (--bench) keeps what an earlier run recorded for the others,
    # whose .out files are still in `out`, if that run used the same model, config and nvcc.
    previous = out / "meta.json"
    if previous.exists() and set(status) != set(BENCHES):
        old = json.loads(previous.read_text())
        if (old.get("config") == values and old.get("model") == model
                and old.get("placement", {}).get("mode") == placement
                and old.get("nvcc_version", nvcc_version) == nvcc_version):
            kept = {bench: entry for bench, entry in old.get("benchmarks", {}).items()
                    if bench not in status and (out / f"{bench}.out").exists()}
            status = {bench: status.get(bench) or kept[bench] for bench in [*BENCHES, *status, *kept]
                      if bench in status or bench in kept}
    meta = {"model": model, "nvcc_version": nvcc_version,
            "config_file": Path(config).name if config else fit.DEFAULT_CONFIG.name,
            "config": values,
            # The active-SM assumption: how it is made here, and the per-SM limits it uses.
            # The number of each configuration is in benchmarks.<bench>.active_sms.<id>.
            "placement": {"mode": placement, "rule": PLACEMENT_RULE[placement], "sm_count": int(values["sm_count"]),
                          "limits": placement_limits(), "hardware_clusters": HARDWARE_CLUSTERS},
            "command": "warplens run --timing --model " + model + " bin/sim/<bench> --sim",
            "benchmarks": status, "not_simulated": NOT_SIMULATED}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    _, rows, _ = fit.read_dir(out)
    table = fit.summarize(rows, "sim_ns")
    count = lambda x: "" if x is None else f"{x:.0f}"
    text = ["# Predicted by the simulator", "",
            f"`{meta['command']}`, config `{meta['config_file']}` (uncalibrated unless stated).", "",
            f"SMs active: how many of the {meta['placement']['sm_count']} SMs the launch is assumed to keep busy; "
            "the model gives each of them that share of the HBM bandwidth. " + PLACEMENT_RULE[placement][0].upper()
            + PLACEMENT_RULE[placement][1:] + ". `meta.json` has the launch shape behind each number. "
            "CTAs per SM is the occupancy limit of the kernel, not how many an SM holds in this launch.", "",
            "| configuration | ns per repetition | PTX warp-instructions per repetition | limited by | SMs active "
            "(assumed) | CTAs per SM (limit) |", "|---|---|---|---|---|---|"]
    for ident, info in table.items():
        row = info["row"]
        placed = f"{count(row.get('sim_active_sms'))} | {count(row.get('sim_ctas_per_sm'))} |"
        if "ns_per_rep" in info:
            text.append(f"| {ident} | {info['ns_per_rep']:.6g} | {info.get('sim_instr_per_rep', float('nan')):.6g} | "
                        f"{row.get('sim_bound', '')} | {placed}")
        else:
            text.append(f"| {ident} (whole launch) | {info['launch_min_ns']:.6g} | | {row.get('sim_bound', '')} | {placed}")
    text += ["", "Not simulated:"] + [f"- `{k}`: {v}" for k, v in NOT_SIMULATED.items()]
    text += [f"- `{b}`: {s['reason']}" for b, s in status.items() if s["status"] != "ok"]
    (out / "predictions.md").write_text("\n".join(text) + "\n")
    return meta


def placement_limits() -> dict:
    """The per-SM limits behind CTAs per SM. Read from the simulator's module when it can be
    imported (it always can where predictions are made)."""
    try:
        from warplens.timing.placement import LIMITS
    except ImportError:
        return {}
    return dict(LIMITS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "predicted")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bench", action="append")
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--model", default="B200")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--placement", choices=PLACEMENTS, default="launch",
                        help="the active-SM assumption: `launch` (default) derives it from each launch's "
                             "shape, `all-sms` assumes every SM is busy in every launch")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--record", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        return one(args.one, args.record, str(args.config) if args.config else None, args.model, args.placement)
    meta = predict(args.out, args.bench, args.config, args.bin, args.model, args.timeout, placement=args.placement)
    return 0 if all(s["status"] == "ok" for s in meta["benchmarks"].values()) else 1


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
