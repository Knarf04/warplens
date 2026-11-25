"""Views of shared memory: which banks each instruction loaded, and where the words of a
logical tile sit physically.

    python -m warplens.viz.smem 32 128 --swizzle 128B            # a TMA tile, as text
    python -m warplens.viz.smem 32 128 --swizzle 128B --svg t.svg
    python -m warplens.viz.smem 32 128 --pitch 132               # a padded tile[32][33] of 4-byte words
    python -m warplens.viz.smem 32 128 --pitch 132 --mark 5 7 --svg t.svg   # and where tile[5][7] is

    from warplens.viz import smem
    svg = smem.bank_heatmap_svg(stats.smem)                      # of a launch

Shared memory has 32 banks of 4 bytes: the word at byte offset o is in bank (o // 4) % 32,
so one 128-byte line of shared memory is one word in each bank. Both views are drawn per
4-byte word.
"""

from __future__ import annotations

import argparse
import sys
from html import escape

import numpy as np

from ..host.tensor_map import SWIZZLE_MODES, swizzle_offsets
from ..units.smem_banks import BANK_BYTES, BANKS, analyze

LINE_BYTES = BANKS * BANK_BYTES  # 128: one word in every bank
_SWIZZLE_BITS = {name.lower(): bits for bits, (name, _) in SWIZZLE_MODES.items()}
# One hue, light to dark, for both views: a count of words, or the bank number.
_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_EMPTY, _INK, _MUTED, _MARK = "#f1f1f1", "#222222", "#666666", "#eb6834"


def _shade(t: float) -> str:
    """The ramp at t in 0..1."""
    x = min(max(t, 0.0), 1.0) * (len(_RAMP) - 1)
    i = min(int(x), len(_RAMP) - 2)
    a, b = (tuple(int(c[k:k + 2], 16) for k in (1, 3, 5)) for c in _RAMP[i:i + 2])
    return "#" + "".join(f"{round(p + (q - p) * (x - i)):02x}" for p, q in zip(a, b))


def _on(t: float) -> str:
    """Ink that is legible on `_shade(t)`."""
    return "#ffffff" if t > 0.45 else _INK


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


# -- which banks each instruction loaded -------------------------------------------------

def _conflict_shade(ratio: float) -> float:
    """Where on the ramp a bank asked for `ratio` times the ideal is: the lightest step for no
    conflict, and a visibly darker one from the first conflict on."""
    return 0.0 if ratio <= 1 else 0.25 + 0.75 * min((ratio - 1) / (BANKS - 1), 1.0)


def bank_heatmap_svg(report, title: str = "Shared-memory banks per instruction") -> str:
    """One row per shared-memory instruction of a launch (`LaunchStats.smem`), one column per
    bank.

    A cell is coloured by conflict, not by load: by the most distinct words one execution put
    in the bank, relative to the wavefronts a conflict-free layout needs for that execution
    (its ideal). A warp that loads 16 bytes per lane puts four words in every bank and needs
    four wavefronts however the data is laid out; its cells are the lightest shade, like those
    of a conflict-free scalar load. A darker cell holds more words than the ideal, and shows
    how many. Beside the grid: the wavefronts the worst execution needed and its ideal, the
    conflict that makes, and the extra wavefronts over all executions. Empty cells were never
    touched. Returns an empty string for a launch without shared-memory accesses."""
    entries = sorted(report.instructions.values(), key=lambda e: (e.kernel, e.line))
    if not entries:
        return ""
    several = len({e.kernel for e in entries}) > 1
    cell, row, left, top = 20, 22, 330, 58
    right = left + BANKS * cell
    columns = (right + 12, right + 150, right + 222)
    width = right + 400
    height = top + len(entries) * row + 64
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           'font-family="sans-serif" font-size="11">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="10" y="18" font-size="14">{escape(title)}</text>',
           f'<text x="{left + BANKS * cell / 2}" y="36" text-anchor="middle" fill="{_MUTED}">bank</text>',
           f'<text x="{columns[0]}" y="{top - 22}" fill="{_MUTED}">wavefronts of the</text>',
           f'<text x="{columns[0]}" y="{top - 8}" fill="{_MUTED}">worst execution</text>',
           f'<text x="{columns[1]}" y="{top - 8}" fill="{_MUTED}">conflict</text>',
           f'<text x="{columns[2]}" y="{top - 22}" fill="{_MUTED}">extra wavefronts,</text>',
           f'<text x="{columns[2]}" y="{top - 8}" fill="{_MUTED}">all executions</text>']
    for bank in range(BANKS):
        if bank % 4 == 0 or bank == BANKS - 1:
            out.append(f'<text x="{left + bank * cell + cell / 2}" y="{top - 8}" text-anchor="middle">{bank}</text>')
    for i, e in enumerate(entries):
        y = top + i * row
        text = " ".join(e.text.split())
        label = f"line {e.line}  {_clip(text, 36)}"
        where = f"{e.kernel}, line {e.line}" if several else f"line {e.line}"
        out.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" font-family="monospace">'
                   f'{escape(label)}<title>{escape(e.kernel)}: {escape(text)}</title></text>')
        for bank in range(BANKS):
            peak, total = int(e.bank_peak[bank]), int(e.bank_words[bank])
            ratio = float(e.bank_ratio[bank])
            t = _conflict_shade(ratio)
            fill = _shade(t) if peak else _EMPTY
            tip = (f"{where}, bank {bank}: at most {peak} distinct word(s) in one execution, "
                   f"{total / e.executions:.2f} on average over {e.executions}; "
                   + ("no more than a conflict-free layout needs" if ratio <= 1 else
                      f"up to {ratio:g} times what a conflict-free layout needs")) if peak else \
                f"{where}, bank {bank}: not touched"
            x = left + bank * cell
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{row - 3}" fill="{fill}">'
                       f'<title>{escape(tip)}</title></rect>')
            if ratio > 1:
                out.append(f'<text x="{x + (cell - 1) / 2}" y="{y + 13}" text-anchor="middle" font-size="10" '
                           f'fill="{_on(t)}" pointer-events="none">{peak}</text>')
        way = "none" if e.worst_way <= 1 else f"{e.worst_way}-way"
        out.append(f'<text x="{columns[0]}" y="{y + 14}">{e.worst} needed, ideal {e.worst_ideal}</text>')
        out.append(f'<text x="{columns[1]}" y="{y + 14}">{way}</text>')
        out.append(f'<text x="{columns[2]}" y="{y + 14}">{e.conflicts:,} in {e.executions:,} '
                   f'execution(s)</text>')
    y = top + len(entries) * row + 10
    out.append(f'<rect x="{left}" y="{y}" width="{cell - 1}" height="12" fill="{_EMPTY}"/>')
    out.append(f'<text x="{left + cell + 4}" y="{y + 10}" fill="{_MUTED}">not touched</text>')
    x0 = left + 100
    out.append(f'<rect x="{x0}" y="{y}" width="{cell - 1}" height="12" fill="{_shade(0)}"/>')
    out.append(f'<text x="{x0 + cell + 4}" y="{y + 10}" fill="{_MUTED}">no conflict: the bank holds no more '
               'words than the ideal number of wavefronts</text>')
    y += 18
    steps = 16
    for k in range(steps):
        ratio = 2 + k * (BANKS - 2) / (steps - 1)
        out.append(f'<rect x="{left + 24 + k * 8}" y="{y}" width="8" height="12" fill="{_shade(_conflict_shade(ratio))}"/>')
    out.append(f'<text x="{left + 19}" y="{y + 10}" text-anchor="end" fill="{_MUTED}">2x</text>')
    out.append(f'<text x="{left + 24 + steps * 8 + 5}" y="{y + 10}" fill="{_MUTED}">{BANKS}x the ideal: a conflict. '
               'The number in the cell is the distinct words in the bank, each needing a wavefront</text>')
    out.append("</svg>")
    return "\n".join(out)


# -- where the words of a tile are --------------------------------------------------------

def swizzle_bits(swizzle) -> int:
    """0 to 3 for None, "32B", "64B", "128B" (or the number itself)."""
    if swizzle is None:
        return 0
    if isinstance(swizzle, (int, np.integer)) and not isinstance(swizzle, bool) and int(swizzle) in SWIZZLE_MODES:
        return int(swizzle)
    bits = _SWIZZLE_BITS.get(str(swizzle).lower())
    if bits is None:
        raise ValueError(f"unknown swizzle {swizzle!r}; choose none, 32B, 64B or 128B")
    return bits


def tile_words(rows: int, row_bytes: int, *, swizzle=None, start: int = 0, pitch: int | None = None) -> np.ndarray:
    """The shared-memory byte offset of every 4-byte word of a logical tile, as an array of
    shape (rows, row_bytes / 4).

    The tile is `rows` rows of `row_bytes` bytes placed from offset `start`. With `swizzle`
    it is placed as TMA places a box (the rows back to back, then each 16-byte chunk moved
    within its 128-byte line, `tensor_map.swizzle_offsets`). Without, `pitch` is the distance
    in bytes from one row to the next, `row_bytes` unless the rows are padded.
    """
    bits = swizzle_bits(swizzle)
    if rows < 1 or row_bytes < BANK_BYTES or row_bytes % BANK_BYTES:
        raise ValueError("a tile needs at least one row, and rows of a multiple of 4 bytes")
    if start < 0 or start % BANK_BYTES:
        raise ValueError("the start offset must be a nonnegative multiple of 4")
    if bits and pitch not in (None, row_bytes):
        raise ValueError("a swizzled tile has its rows back to back; --pitch is for an unswizzled one")
    if bits and (start % 16 or row_bytes % 16):
        raise ValueError("a swizzled tile starts at, and has rows of, a multiple of 16 bytes")
    pitch = row_bytes if pitch is None else pitch
    if pitch < row_bytes or pitch % BANK_BYTES:
        raise ValueError("the pitch must be a multiple of 4 that is at least the row size")
    linear = (start + np.arange(rows, dtype=np.uint64)[:, None] * np.uint64(pitch)
              + np.arange(0, row_bytes, BANK_BYTES, dtype=np.uint64)[None, :])
    return swizzle_offsets(linear, bits).astype(np.int64)


def _check_words(words) -> np.ndarray:
    words = np.asarray(words, dtype=np.int64)
    if words.ndim != 2 or not words.size or (words < 0).any() or (words % BANK_BYTES).any():
        raise ValueError("expected a 2-D array of word offsets, each a nonnegative multiple of 4")
    return words


def column_conflict(words) -> int:
    """N for the worst N-way bank conflict of a warp that reads one word per lane down a
    column of the tile, 32 consecutive rows at a time (1 = none)."""
    words = _check_words(words)
    worst = 1
    for first in range(0, words.shape[0], 32):
        for column in words[first:first + 32].T:
            waves, ideal, _ = analyze(column, BANK_BYTES)
            worst = max(worst, -(-waves // ideal))
    return worst


def _conflict_text(words) -> str:
    way = column_conflict(words)
    return "a warp reading down a column, one word per lane, has " + \
        ("no bank conflict" if way <= 1 else f"a {way}-way bank conflict")


def layout_text(words, title: str = "Shared-memory layout of a tile", max_rows: int = 16) -> str:
    """The bank of every word of a logical tile (`tile_words`), one line per row."""
    words = _check_words(words)
    rows, columns = words.shape
    lines = [title,
             f"  {rows} rows of {columns} words (4 bytes each) at shared offsets "
             f"{int(words.min())}-{int(words.max()) + BANK_BYTES - 1}; {_conflict_text(words)}",
             "  row  offset of its first word  bank of each word"]
    for r in range(min(rows, max_rows)):
        banks = " ".join(f"{int(o) // BANK_BYTES % BANKS:>2}" for o in words[r])
        lines.append(f"  {r:>3}  {int(words[r, 0]):>24}  {banks}")
    if rows > max_rows:
        lines.append(f"  ... {rows - max_rows} more row(s)")
    return "\n".join(lines)


def layout_svg(words, title: str = "Shared-memory layout of a tile", max_rows: int = 32,
               mark: tuple[int, int] | None = None) -> str:
    """Two pictures of a logical tile (`tile_words`), per 4-byte word and coloured by bank.

    Left, the tile as the program indexes it: a column that is one shade puts every row's
    word in the same bank, which a warp reading down the column pays for. Right, shared
    memory itself, 128 bytes (one word per bank) per line, with the tile's words where they
    physically are. The first word of each row is outlined in both, and each cell's tooltip
    names the other side. At most `max_rows` rows are drawn.

    `mark` is the (row, word within the row) of one element to follow from the left picture
    to the right: it is ringed in both, and a line under the pictures says where it is.
    """
    words = _check_words(words)
    rows, columns = words.shape
    shown = words[:max_rows]
    first_line = int(shown.min()) // LINE_BYTES
    lines = int(shown.max()) // LINE_BYTES - first_line + 1
    cut = lines > 3 * max_rows  # rows very far apart: do not draw the gap
    lines = min(lines, 3 * max_rows)
    cell = 14 if columns <= 32 else max(448 // columns, 4)
    h = 14
    left, top, gap = 64, 86, 96
    logical_w = columns * cell
    px = left + max(logical_w, 220) + gap  # x of the physical panel
    width = px + BANKS * 14 + 20
    height = top + max(len(shown), lines) * h + 44 + (14 if mark is not None else 0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           'font-family="sans-serif" font-size="11">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="10" y="18" font-size="14">{escape(title)}</text>',
           f'<text x="10" y="36" fill="{_MUTED}">{rows} rows of {columns} words of 4 bytes; '
           f'{escape(_conflict_text(words))}</text>',
           f'<text x="{left}" y="58" font-size="12">Logical tile: row, byte within the row</text>',
           f'<text x="{px}" y="58" font-size="12">Shared memory: offset of each 128-byte line, bank</text>']
    for c in range(columns):
        if (c * BANK_BYTES) % 16 == 0 and (cell >= 14 or c % 8 == 0):
            out.append(f'<text x="{left + c * cell}" y="{top - 6}" fill="{_MUTED}" font-size="9">{c * BANK_BYTES}</text>')
    for bank in range(BANKS):
        if bank % 4 == 0 or bank == BANKS - 1:
            out.append(f'<text x="{px + bank * 14 + 7}" y="{top - 6}" text-anchor="middle" fill="{_MUTED}" '
                       f'font-size="9">{bank}</text>')
    for r in range(len(shown)):
        out.append(f'<text x="{left - 6}" y="{top + r * h + 11}" text-anchor="end" fill="{_MUTED}">{r}</text>')
    for line in range(lines):
        out.append(f'<text x="{px - 6}" y="{top + line * h + 11}" text-anchor="end" fill="{_MUTED}">'
                   f'{(first_line + line) * LINE_BYTES}</text>')
        out.append(f'<rect x="{px}" y="{top + line * h}" width="{BANKS * 14 - 1}" height="{h - 1}" fill="{_EMPTY}"/>')
    outlined = []
    for r in range(len(shown)):
        for c in range(columns):
            offset = int(shown[r, c])
            bank = offset // BANK_BYTES % BANKS
            fill = _shade(bank / (BANKS - 1))
            tip = (f"row {r}, bytes {c * BANK_BYTES}-{c * BANK_BYTES + 3}: shared offset {offset}, bank {bank}")
            x, y = left + c * cell, top + r * h
            out.append(f'<rect x="{x}" y="{y}" width="{max(cell - 1, 1)}" height="{h - 1}" fill="{fill}">'
                       f'<title>{tip}</title></rect>')
            line = offset // LINE_BYTES - first_line
            if line < lines:
                x2, y2 = px + bank * 14, top + line * h
                out.append(f'<rect x="{x2}" y="{y2}" width="13" height="{h - 1}" fill="{fill}">'
                           f'<title>{tip}</title></rect>')
                if c == 0:
                    outlined.append((x2, y2, 13))
            if c == 0:
                outlined.append((x, y, max(cell - 1, 1)))
    for x, y, w in outlined:  # last, so that no neighbour paints over them
        out.append(f'<rect x="{x + .75}" y="{y + .75}" width="{w - 1.5}" height="{h - 2.5}" fill="none" '
                   f'stroke="{_INK}" stroke-width="1.5" pointer-events="none"/>')
    notes = ["Colour is the bank, light (0) to dark (31). Outlined: the first word of each row. "
             "Grey: not part of the tile."]
    if mark is not None:
        r, c = mark
        if not (0 <= r < rows and 0 <= c < columns):
            raise ValueError(f"mark {mark} is outside a tile of {rows} rows of {columns} words")
        offset = int(words[r, c])
        bank, line = offset // BANK_BYTES % BANKS, offset // LINE_BYTES - first_line
        spots = [(left + c * cell, top + r * h, max(cell - 1, 1))] if r < len(shown) else []
        if r < len(shown) and line < lines:
            spots.append((px + bank * 14, top + line * h, 13))
        for x, y, w in spots:  # a ring that shows on the lightest and on the darkest bank
            for colour, stroke in (("#ffffff", 5), (_MARK, 2.5)):
                out.append(f'<rect x="{x - 2}" y="{y - 2}" width="{w + 4}" height="{h + 3}" fill="none" '
                           f'stroke="{colour}" stroke-width="{stroke}" pointer-events="none"/>')
        notes.append(f"Ringed: row {r}, bytes {c * BANK_BYTES}-{c * BANK_BYTES + 3} of the tile is at shared "
                     f"offset {offset}: the line at {offset // LINE_BYTES * LINE_BYTES}, bank {bank}.")
    y = top + max(len(shown), lines) * h + 14
    if rows > len(shown) or cut:
        notes.append(f"The first {len(shown)} of {rows} rows are drawn"
                     + ("; lines beyond the picture are left out." if cut else "."))
    for k, note in enumerate(notes):
        out.append(f'<text x="10" y="{y + 14 * k}" fill="{_MUTED}">{note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def tile_title(rows: int, row_bytes: int, swizzle=None, start: int = 0, pitch: int | None = None) -> str:
    bits = swizzle_bits(swizzle)
    how = f"TMA swizzle {SWIZZLE_MODES[bits][0]}" if bits else \
        ("no swizzle" if pitch in (None, row_bytes) else f"rows {pitch} bytes apart")
    return f"Tile of {rows} rows x {row_bytes} bytes at shared offset {start}, {how}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m warplens.viz.smem", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=int, help="rows of the tile")
    parser.add_argument("row_bytes", type=int, help="bytes per row, a multiple of 4")
    parser.add_argument("--swizzle", default="none", help="place it as TMA does: none, 32B, 64B or 128B")
    parser.add_argument("--start", type=int, default=0, help="shared-memory offset of the tile")
    parser.add_argument("--pitch", type=int, help="bytes from one row to the next when the rows are "
                                                  "padded (unswizzled only). Default: the row size")
    parser.add_argument("--mark", type=int, nargs=2, metavar=("ROW", "WORD"),
                        help="ring one element in the SVG and say where it is: its row, and its 4-byte "
                             "word within the row")
    parser.add_argument("--svg", metavar="FILE", help="write the picture as SVG")
    args = parser.parse_args(argv)
    try:
        words = tile_words(args.rows, args.row_bytes, swizzle=args.swizzle, start=args.start, pitch=args.pitch)
        title = tile_title(args.rows, args.row_bytes, args.swizzle, args.start, args.pitch)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mark = None if args.mark is None else tuple(args.mark)
    if mark is not None and not (0 <= mark[0] < words.shape[0] and 0 <= mark[1] < words.shape[1]):
        print(f"error: --mark {mark[0]} {mark[1]} is outside the tile", file=sys.stderr)
        return 2
    print(layout_text(words, title))
    if mark is not None:
        offset = int(words[mark])
        print(f"  row {mark[0]}, word {mark[1]} is at shared offset {offset}, bank {offset // BANK_BYTES % BANKS}")
    if args.svg:
        with open(args.svg, "w") as f:
            f.write(layout_svg(words, title, mark=mark))
        print(f"wrote {args.svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
