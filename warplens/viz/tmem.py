"""Views of tensor memory: where an MMA puts its result, and which cells each
thread's registers reach in a tcgen05.ld / tcgen05.st.

    python -m warplens.viz.tmem 64 32                 # accumulator layout as text
    python -m warplens.viz.tmem 64 32 --svg out.svg   # the same as a picture
    python -m warplens.viz.tmem --transfer 16x128b 2  # a load/store shape

TMEM has 128 lanes; warp w of a CTA can reach lanes 32 * (w % 4) .. + 31.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..units.tcgen05 import LANES, _accumulator_lanes, transfer_cells


def accumulator_rows(m_dim: int, lane_offset: int = 0) -> np.ndarray:
    """For each of the 128 lanes, the result row of a `cta_group::1` MMA stored
    there, or -1 if the MMA leaves the lane untouched.

    `lane_offset` is the lane part of the MMA's TMEM address. For M = 64 (PTX's
    "Layout F") it is 0 or 16 and selects the low or high 16 lanes of each
    warp's 32, so two 64-row results can share the same columns.
    """
    if m_dim not in (64, 128):
        raise ValueError("M must be 64 or 128 for cta_group::1")
    if lane_offset not in ((0, 16) if m_dim == 64 else (0,)):
        raise ValueError("the lane offset must be 0, or 16 for M = 64")
    rows = np.full(LANES, -1)
    rows[lane_offset + _accumulator_lanes(m_dim)] = np.arange(m_dim)
    return rows


def locate(m: int, n: int, m_dim: int, n_dim: int, pair: bool = False, lane_offset: int = 0):
    """Where element (m, n) of an MMA result lives: (CTA within the pair, lane, column).

    For `cta_group::1` the CTA is always 0. For `cta_group::2` the even-ranked CTA holds
    the first half of the rows and the odd-ranked one the second half. With M = 256 each
    holds 128 rows on its 128 lanes. With M = 128 (PTX's "Layout B") each holds 64 rows,
    which fill only half the lanes, so the columns are folded: the first N/2 columns sit
    on lanes 0-63 and the rest on lanes 64-127, in N/2 TMEM columns.
    """
    if not (0 <= m < m_dim and 0 <= n < n_dim):
        raise ValueError(f"({m}, {n}) is outside a {m_dim} x {n_dim} result")
    if not pair:
        return 0, int(accumulator_rows(m_dim, lane_offset).tolist().index(m)), n
    if m_dim not in (128, 256):
        raise ValueError("M must be 128 or 256 for cta_group::2")
    if not 16 <= n_dim <= 256 or n_dim % 16:
        raise ValueError("N must be a multiple of 16 from 16 to 256 for cta_group::2")
    if lane_offset != 0:
        raise ValueError("the lane offset must be 0 for cta_group::2")
    half = m_dim // 2
    cta, row = divmod(m, half)
    if m_dim == 256:
        return cta, row, n
    return cta, row + 64 * (n >= n_dim // 2), n % (n_dim // 2)


def pair_text(m_dim: int, n_dim: int) -> str:
    """The TMEM layout of a cta_group::2 MMA result, for both CTAs of the pair."""
    locate(0, 0, m_dim, n_dim, pair=True)  # validates M
    half = m_dim // 2
    lines = [f"TMEM layout of a {m_dim} x {n_dim} tcgen05.mma result (cta_group::2"
             + (", Layout B)" if m_dim == 128 else ")"),
             "  CTA    lanes      holds"]
    for cta, name in enumerate(("even", "odd ")):
        rows = f"rows {cta * half}-{cta * half + half - 1}"
        if m_dim == 256:
            lines.append(f"  {name}   0-127      {rows}, columns 0-{n_dim - 1} in TMEM columns 0-{n_dim - 1}")
        else:
            lines.append(f"  {name}   0-63       {rows}, columns 0-{n_dim // 2 - 1} in TMEM columns 0-{n_dim // 2 - 1}")
            lines.append(f"  {name}   64-127     {rows}, columns {n_dim // 2}-{n_dim - 1} in TMEM columns 0-{n_dim // 2 - 1}")
    return "\n".join(lines)


def _runs(rows: np.ndarray):
    """Consecutive lanes holding consecutive rows (or all unused), as (first lane, last lane, first row)."""
    start = 0
    for lane in range(1, LANES + 1):
        if lane == LANES or lane % 32 == 0 or (rows[lane] < 0) != (rows[start] < 0) \
                or (rows[lane] >= 0 and rows[lane] != rows[lane - 1] + 1):
            yield start, lane - 1, int(rows[start])
            start = lane


def _title(m_dim: int, n_dim: int, lane_offset: int) -> str:
    extra = (f", Layout F, lane offset {lane_offset}" if m_dim == 64 else "")
    return f"TMEM layout of a {m_dim} x {n_dim} tcgen05.mma result (cta_group::1{extra})"


def accumulator_text(m_dim: int, n_dim: int, lane_offset: int = 0) -> str:
    rows = accumulator_rows(m_dim, lane_offset)
    lines = [_title(m_dim, n_dim, lane_offset),
             f"  result column n is TMEM column n: {n_dim} columns of 32 bits",
             "  lanes      warp  holds"]
    for first, last, row in _runs(rows):
        held = f"result rows {row}-{row + last - first}" if row >= 0 else "not written"
        lines.append(f"  {first:>3}-{last:<3}    {first // 32}     {held}")
    return "\n".join(lines)


def accumulator_svg(m_dim: int, n_dim: int, lane_offset: int = 0) -> str:
    rows = accumulator_rows(m_dim, lane_offset)
    cell, left, top, width = 5, 150, 40, 360
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 190}" '
           f'height="{top + LANES * cell + 30}" font-family="sans-serif" font-size="12">',
           f'<text x="10" y="20" font-size="14">{_title(m_dim, n_dim, lane_offset)}</text>']
    for first, last, row in _runs(rows):
        y, h = top + first * cell, (last - first + 1) * cell
        fill = colors[first // 32] if row >= 0 else "#eeeeee"
        label = f"rows {row}-{row + last - first}" if row >= 0 else "not written"
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{h}" fill="{fill}" '
                   f'fill-opacity="{0.85 if row >= 0 else 1}" stroke="#ffffff"/>')
        out.append(f'<text x="{left + width / 2}" y="{y + h / 2 + 4}" text-anchor="middle" '
                   f'fill="{"#ffffff" if row >= 0 else "#777777"}">{label}</text>')
        out.append(f'<text x="{left - 8}" y="{y + h / 2 + 4}" text-anchor="end">lanes {first}-{last}</text>')
    for warp in range(4):
        y = top + warp * 32 * cell
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{32 * cell}" fill="none" stroke="#333333"/>')
        out.append(f'<text x="{left + width + 10}" y="{y + 16 * cell + 4}">reachable by warp {warp} (mod 4)</text>')
    out.append(f'<text x="{left + width / 2}" y="{top + LANES * cell + 20}" text-anchor="middle">'
               f'columns 0-{n_dim - 1}: result column n is TMEM column n</text>')
    out.append("</svg>")
    return "\n".join(out)


def transfer_text(shape: str, count: int) -> str:
    lanes, columns = transfer_cells(shape, count)
    registers = lanes.shape[1]
    lines = [f"tcgen05.ld/st.{shape}.x{count}: {registers} register(s) per thread, "
             f"{lanes.max() + 1} lanes x {columns.max() + 1} columns from the address",
             "  thread  " + "  ".join(f"reg {q} (lane,col)".ljust(16) for q in range(registers))]
    for thread in range(32):
        cells = "  ".join(f"({lanes[thread, q]:>2},{columns[thread, q]:>3})".ljust(16) for q in range(registers))
        lines.append(f"  {thread:>6}  {cells}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m warplens.viz.tmem", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("m", nargs="?", type=int, default=128, help="MMA rows: 64 or 128")
    parser.add_argument("n", nargs="?", type=int, default=64, help="MMA columns")
    parser.add_argument("--lane-offset", type=int, default=0,
                        help="lane part of the MMA's TMEM address: 0, or 16 for M = 64")
    parser.add_argument("--pair", action="store_true",
                        help="a cta_group::2 MMA over two CTAs (M = 128 or 256)")
    parser.add_argument("--svg", metavar="FILE", help="write the accumulator layout as SVG")
    parser.add_argument("--transfer", nargs=2, metavar=("SHAPE", "COUNT"),
                        help="show a load/store shape instead, e.g. 16x128b 2")
    args = parser.parse_args(argv)
    try:
        if args.transfer:
            print(transfer_text(args.transfer[0], int(args.transfer[1])))
            return 0
        if args.pair:
            if args.lane_offset:
                raise ValueError("the lane offset must be 0 for cta_group::2")
            if args.svg:
                raise ValueError("SVG output is not implemented for cta_group::2")
            print(pair_text(args.m, args.n))
            return 0
        print(accumulator_text(args.m, args.n, args.lane_offset))
        if args.svg:
            with open(args.svg, "w") as f:
                f.write(accumulator_svg(args.m, args.n, args.lane_offset))
            print(f"wrote {args.svg}")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
