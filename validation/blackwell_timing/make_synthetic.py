#!/usr/bin/env python3
"""Make a SYNTHETIC results directory, to prove fit.py works without a Blackwell GPU.

    PYTHONPATH=<repo> python make_synthetic.py OUT_DIR [--small]

Nothing here is a measurement. The "hardware" is the simulator itself running with a
different, known set of parameters (TRUTH below). Each simulated launch becomes a BENCH
line as the hardware build would print it: the kernel time extrapolated linearly to
1000 times the repetitions, plus a 20 us launch overhead, plus (for every trial but the
first) up to 0.4% of positive noise. fit.py, given these lines and the committed
predictions of the unmodified config, must recover TRUTH. tests/test_timing_calibration.py
checks that on the committed copy in fixture_synthetic/ (made with --small).

Every output line and file says it is synthetic: DEVICE has `synthetic=1` and
`name=SYNTHETIC_not_measured`, and fit.py then labels everything it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
import predict  # noqa: E402

# The parameters of the pretend hardware. Chosen to differ from b200.yaml in every entry a
# benchmark calibrates, with the TMA engine slower than L2 so that its rate is identifiable.
TRUTH = {
    "clock_ghz": 1.9,
    "hbm_bandwidth_tb_s": 3.0,
    "l2_bandwidth_per_sm_gb_s": 300.0,
    "tma_latency_ns": 240.0,
    "tma_bytes_per_clk": 60.0,
    "smem_wavefront_clk": 1.6,
    "global_access_latency_clk": 45.0,
    "tmem_access_clk": 7.0,
    "tensor_core_tflops_f16": 1500.0,
    "tensor_core_tflops_tf32": 800.0,
    "mma_latency_clk": 20.0,
}
LAUNCH_NS = 20000.0
REPS_FACTOR = 1000
SMALL = ("launch/", "alu_chain/", "smem_wavefronts/load_chain", "global_load/", "gstream/", "cp_async/",
         "mbar/", "tma/", "m=128/n=256", "tmem_access/")


def truth_config(path: Path) -> None:
    import yaml
    config = yaml.safe_load(fit.DEFAULT_CONFIG.read_text())
    for key, val in TRUTH.items():
        config[key]["value"] = val
        config[key]["source"] = "assumption: parameter of the SYNTHETIC hardware in make_synthetic.py"
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def noise(ident: str, trial: int, reps: float) -> float:
    if trial == 0:
        return 0.0
    return (zlib.crc32(f"{ident}/{trial}/{reps}".encode()) % 1000) / 1000 * 0.004


def synthesize(out: Path, small: bool = False, trials: int = 3) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        truth_config(scratch / "truth.yaml")
        meta = predict.predict(scratch / "sim", config=scratch / "truth.yaml", quiet=True)
        failed = {b: s for b, s in meta["benchmarks"].items() if s["status"] != "ok"}
        if failed:
            raise SystemExit(f"the simulator did not run: {failed}")
        _, rows, _ = fit.read_dir(scratch / "sim")
    table = fit.summarize(rows, "sim_ns")
    lines: dict[str, list[str]] = {}
    for ident, info in table.items():
        if small and not any(s in ident for s in SMALL):
            continue
        first = info["row"]
        extra = " ".join(f"{k}={first[k]:g}" if isinstance(first[k], float) else f"{k}={first[k]}"
                         for k in first if k not in ("id", "reps", "trial", "event_ns", "wall_ns", "clk", "check",
                                                     "file") and not k.startswith("sim_"))
        bench = lines.setdefault(info["bench"], [])
        mine = [r for r in rows if r["id"] == ident]
        if "ns_per_rep" in info:
            slope, intercept = info["ns_per_rep"] * info["norm"], info["intercept_ns"]
            if ident.startswith("gstream/"):  # the model has no bandwidth limit for ordinary accesses
                slope, intercept = 24 / (TRUTH["hbm_bandwidth_tb_s"] * 1000), 0.0
            points = sorted({r["reps"] for r in mine})
            factor = 1 if ident.startswith("gstream/") else REPS_FACTOR
            points = [x * factor * (65536 if ident.startswith("gstream/") else 1) for x in points]
        else:
            slope, intercept, points = 0.0, info["launch_min_ns"], [0.0]
        for trial in range(trials):
            for reps in points:
                kernel = intercept + slope * reps
                event = LAUNCH_NS + kernel * (1 + noise(ident, trial, reps))
                bench.append(f"BENCH id={ident} {extra + ' ' if extra else ''}reps={reps:.0f} trial={trial} "
                             f"event_ns={event:.3f} wall_ns={event + 5000:.3f} "
                             f"clk={kernel * TRUTH['clock_ghz']:.0f} check=0")
    for bench, body in lines.items():
        device = (f"DEVICE bench={bench} name=SYNTHETIC_not_measured cc=10.0 sm_count=148 "
                  f"total_mem_bytes=196000000000 free_mem_bytes=196000000000 bench_clock=1 sim=0 "
                  f"trials={trials} scale=1 synthetic=1")
        (out / f"{bench}.out").write_text("\n".join([device] + body) + "\n")
    (out / "truth.json").write_text(json.dumps(TRUTH, indent=1) + "\n")
    (out / "SYNTHETIC.txt").write_text(
        "SYNTHETIC data written by validation/blackwell_timing/make_synthetic.py.\n"
        "Not measured on any GPU. The 'hardware' is the simulator run with the parameters in\n"
        "truth.json; it exists to test fit.py.\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--small", action="store_true", help="fewer configurations and two trials (the test fixture)")
    args = parser.parse_args(argv)
    synthesize(args.out, args.small, trials=2 if args.small else 3)
    print(f"wrote SYNTHETIC results to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
