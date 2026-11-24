#!/usr/bin/env python3
"""Run every timing microbenchmark in the simulator and record what the model predicts.

    PYTHONPATH=<repo> python predict.py [--out predicted] [--config YAML] [--bench NAME ...]

Each program in bin/sim (build.sh sim) is run as `warplens run --timing --model B200
bin/sim/<bench> --sim`. The command is called in-process only so that the launch records
can be read with full precision; the summary it prints rounds to whole nanoseconds.

Writes into --out:
  <bench>.out   the program's DEVICE and BENCH lines, each BENCH line with the simulator's
                numbers for that launch appended: sim_ns (kernel time), sim_instr
                (warp-instructions), sim_bound (the model's bottleneck verdict)
  meta.json     the config values used, and per benchmark whether the simulator ran it
                (if not, why). With --bench, the entries of the other benchmarks are kept
                from the meta.json already there, if it was made the same way
  predictions.md  a table of ns per repetition per configuration

--model selects the model and its default config (B200 unless specified).
--config runs with another timing config in place of that default, e.g.
the calibrated.yaml that fit.py writes, to check that the calibrated model reproduces
the measurements.
"""

from __future__ import annotations

import argparse
import json
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


def one(program: str, record: str, config: str | None, model: str) -> int:
    """Child process: `warplens run --timing program --sim`, launch records to `record`."""
    from warplens import cli
    if config:
        import shutil
        from warplens.timing import config as timing_config
        scratch = Path(tempfile.mkdtemp(prefix="warplens_timing_config_"))
        shutil.copy(config, scratch / f"{model.lower()}.yaml")
        timing_config.CONFIG_DIR = scratch
    launches = []

    def capture(done, _model):
        launches.extend(done)

    cli.summary = capture
    code = cli.main(["run", "--timing", "--model", model, program, "--sim"])
    Path(record).write_text(json.dumps([
        {"kernel": name, "ctas": stats.ctas, "instructions": stats.instructions,
         "time_ns": stats.timing.time_ns, "bound": stats.timing.bottleneck.replace(" ", "_"),
         "findings": len(stats.findings)} for name, stats in launches]))
    return code


def predict(out: Path, benches=None, config: Path | None = None, bin_dir: Path | None = None,
            model: str = "B200", timeout: int = 1800, quiet: bool = False) -> dict:
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
            command = [sys.executable, str(Path(__file__).resolve()), "--one", str(program), "--record", str(record), "--model", model]
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
        lines, at = [], 0
        for line in stdout.splitlines():
            if line.startswith("BENCH "):
                if at < len(launches):
                    sim = launches[at]
                    tokens = [t if not t.startswith("wall_ns=") else "wall_ns=0" for t in line.split()]
                    line = " ".join(tokens) + (f" sim_ns={sim['time_ns']!r} sim_instr={sim['instructions']}"
                                               f" sim_bound={sim['bound']} sim_findings={sim['findings']}")
                at += 1
                lines.append(line)
            elif line.startswith(("DEVICE ", "BENCH_ERROR ")):
                lines.append(line)
        (out / f"{bench}.out").write_text("\n".join(lines) + "\n")
        if code == 0 and at == len(launches) and at:
            status[bench] = {"status": "ok", "launches": at}
        else:
            message = [l for l in stderr.splitlines() if l.strip()]
            status[bench] = {"status": "failed", "returncode": code, "launches": min(at, len(launches)),
                             "reason": " | ".join(message[:4]) or f"{at} BENCH lines for {len(launches)} launches"}
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
                and old.get("nvcc_version", nvcc_version) == nvcc_version):
            kept = {bench: entry for bench, entry in old.get("benchmarks", {}).items()
                    if bench not in status and (out / f"{bench}.out").exists()}
            status = {bench: status.get(bench) or kept[bench] for bench in [*BENCHES, *status, *kept]
                      if bench in status or bench in kept}
    meta = {"model": model, "nvcc_version": nvcc_version,
            "config_file": Path(config).name if config else fit.DEFAULT_CONFIG.name,
            "config": values, "active_sms": int(values["sm_count"]),
            "command": "warplens run --timing --model " + model + " bin/sim/<bench> --sim",
            "benchmarks": status, "not_simulated": NOT_SIMULATED}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    _, rows, _ = fit.read_dir(out)
    table = fit.summarize(rows, "sim_ns")
    text = ["# Predicted by the simulator", "",
            f"`{meta['command']}`, config `{meta['config_file']}` (uncalibrated unless stated), "
            f"{meta['active_sms']} SMs assumed to share HBM.", "",
            "| configuration | ns per repetition | PTX warp-instructions per repetition | limited by |", "|---|---|---|---|"]
    for ident, info in table.items():
        if "ns_per_rep" in info:
            text.append(f"| {ident} | {info['ns_per_rep']:.6g} | {info.get('sim_instr_per_rep', float('nan')):.6g} | "
                        f"{info['row'].get('sim_bound', '')} |")
        else:
            text.append(f"| {ident} (whole launch) | {info['launch_min_ns']:.6g} | | {info['row'].get('sim_bound', '')} |")
    text += ["", "Not simulated:"] + [f"- `{k}`: {v}" for k, v in NOT_SIMULATED.items()]
    text += [f"- `{b}`: {s['reason']}" for b, s in status.items() if s["status"] != "ok"]
    (out / "predictions.md").write_text("\n".join(text) + "\n")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "predicted")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bench", action="append")
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--model", default="B200")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--record", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        return one(args.one, args.record, str(args.config) if args.config else None, args.model)
    meta = predict(args.out, args.bench, args.config, args.bin, args.model, args.timeout)
    return 0 if all(s["status"] == "ok" for s in meta["benchmarks"].values()) else 1


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
