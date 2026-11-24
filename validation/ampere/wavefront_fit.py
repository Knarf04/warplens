"""Attribute the shared-memory wavefront offset (validation matrix TIM-17).

    python wavefront_fit.py wavefront_probe_ncu.csv wavefront_probe_configs.csv

For every launch of bin/wavefront_probe, the model predicts

    wavefronts_mem_shared = shared loads + shared stores + bank conflicts

(the probe never conflicts, so the last term should be 0). The offset is what
ncu reports above that. This prints the offset per launch and a least-squares
fit of it to the quantities the probe varies one at a time.
"""

import csv
import sys
from pathlib import Path

import numpy as np

P = "smsp__inst_executed_op_"
WF = "l1tex__data_pipe_lsu_wavefronts"
CONF = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared"


def read_ncu(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        sys.exit(f"{path}: no ncu CSV table found. ncu output was:\n" + "\n".join(lines[:20]))
    rows: dict[int, dict] = {}
    for row in csv.DictReader(lines[start:]):
        r = rows.setdefault(int(row["ID"]), {"kernel": row["Kernel Name"].split("(")[0],
                                             "block": row["Block Size"], "grid": row["Grid Size"]})
        try:
            r[row["Metric Name"].removesuffix(".sum")] = float(row["Metric Value"].replace(",", ""))
        except ValueError:
            pass  # "n/a"
    return [rows[i] for i in sorted(rows)]


def main() -> int:
    hw = read_ncu(Path(sys.argv[1]))
    cfg = list(csv.DictReader(Path(sys.argv[2]).open()))
    if len(hw) != len(cfg):
        sys.exit(f"{len(hw)} profiled launches but {len(cfg)} configurations")

    names = ["ctas", "warps", "sh_st", "sh_ld", "gl_ld", "gl_st", "cta_barriers", "warp_barriers"]
    feats, offsets = [], []
    print(f"{'#':>2} {'kernel':<11} {'grid':>4} {'block':>7} {'st ld gld gst bar':<17} "
          f"{'sh_st':>6} {'sh_ld':>6} {'conf':>5} {'wf_sh':>7} {'offset':>7} {'/cta':>7} {'/warp':>6} "
          f"{'wf_lg':>7} {'wf_all':>7}")
    for h, c in zip(hw, cfg):
        grid, bx, by = int(c["grid"]), int(c["bx"]), int(c["by"])
        if h["kernel"] != c["kernel"] or h["block"] != f"({bx}, {by}, 1)" or h["grid"] != f"({grid}, 1, 1)":
            sys.exit(f"launch {c['launch']}: ncu row {h['kernel']} {h['grid']} {h['block']} does not match {c}")
        warps = grid * -(-bx * by // 32)
        n = {k: int(c[k]) for k in ("n_st", "n_ld", "n_gld", "n_gst", "n_bar")}
        # Instruction counts as measured where ncu has them, else as configured.
        sh_st = h.get(P + "shared_st", warps * n["n_st"])
        sh_ld = h.get(P + "shared_ld", warps * n["n_ld"])
        gl_ld = h.get(P + "global_ld", warps * n["n_gld"])
        gl_st = h.get(P + "global_st", warps * n["n_gst"])
        conf = h.get(CONF, 0.0)
        wf = h.get(WF + "_mem_shared")
        if wf is None:
            sys.exit(f"launch {c['launch']}: no {WF}_mem_shared value")
        off = wf - sh_st - sh_ld - conf
        feats.append([grid, warps, sh_st, sh_ld, gl_ld, gl_st, grid * n["n_bar"], warps * n["n_bar"]])
        offsets.append(off)
        print(f"{c['launch']:>2} {c['kernel']:<11} {grid:>4} {f'{bx}x{by}':>7} "
              f"{n['n_st']:>2} {n['n_ld']:>2} {n['n_gld']:>3} {n['n_gst']:>3} {n['n_bar']:>3}   "
              f"{sh_st:>6.0f} {sh_ld:>6.0f} {conf:>5.0f} {wf:>7.0f} {off:>+7.0f} {off / grid:>7.2f} {off / warps:>6.2f} "
              f"{h.get(WF + '_mem_lg', float('nan')):>7.0f} {h.get(WF, float('nan')):>7.0f}")

    a, y = np.array(feats, dtype=float), np.array(offsets)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    print("\nleast-squares fit: offset = " + " + ".join(f"{c:.3f}*{n}" for c, n in zip(coef, names)))
    print(f"largest residual {np.abs(resid).max():.2f} wavefronts (launch {int(np.abs(resid).argmax())}), "
          f"rms {np.sqrt((resid ** 2).mean()):.2f}")
    for label, cols in (("one per CTA", [0]), ("one per warp", [1]), ("one per shared access", [2, 3]),
                        ("one per shared access plus one per CTA", [0, 2, 3]),
                        ("one per shared access plus one per warp", [1, 2, 3]),
                        ("one per global access plus one per CTA", [0, 4, 5]),
                        ("two per warp plus one per CTA", [0, 1, 1]),
                        ("two per warp plus one per CTA per barrier", [1, 1, 6]),
                        ("two per warp plus one per warp per barrier", [1, 1, 7])):
        r = y - a[:, cols].sum(axis=1)
        print(f"  hypothesis {label + ':':<44} wrong in {int((r != 0).sum()):>2} of {len(y)} launches, "
              f"largest error {np.abs(r).max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
