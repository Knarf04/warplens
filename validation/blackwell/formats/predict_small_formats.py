"""What formats/small_formats.cu should print, according to warplens.numerics.

    python validation/blackwell/formats/predict_small_formats.py fp8|fp4|e8m0|special

The simulator does not execute the conversion instructions, so this stands in for
`warplens run` for that probe. The input generators below mirror the ones in
small_formats.cu and must be kept in step with them. Lines the probe prints with an
"info:" prefix (inputs warplens.numerics refuses) are left out; compare.py ignores them.
"""

import struct
import sys

import numpy as np

from warplens.numerics import E2M1, E4M3, E5M2, UE8M0


def sweep_inputs():
    values = []
    for sign in range(2):
        for e in range(-18, 18):
            for j in range(32):
                base = (sign << 31) | ((e + 127) << 23) | (j << 18)
                values += [(base - 1) & 0xFFFFFFFF, base, base + 1]
    return values


def e8m0_inputs():
    fractions = [0x000000, 0x000001, 0x3504F3, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF]
    return [(e << 23) | fraction for e in range(1, 255) for fraction in fractions]


def special_inputs():
    values = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF,
              0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFA12345]
    for v in [1e10, 448.0, 464.0, 479.9, 480.0, 512.0, 57344.0, 61440.0, 65536.0, 6.0, 6.9, 7.0, 8.0,
              100.0, 0.25, 0.24, 1e-3, 1e-10, 1e-30, 1e30, 3.0e38, 1.0, 0.5, 0.75]:
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        values += [bits, bits | 0x80000000]
    return values


def encoding(name, fmt, inputs):
    lines, row = [], []
    for bits in inputs:
        x = float(np.array([bits], dtype=np.uint32).view(np.float32)[0])
        nan = x != x
        negative = bool(bits >> 31)
        if (fmt is E2M1 and nan) or (fmt is UE8M0 and negative and x != 0 and not nan):
            if row:  # the probe breaks the line before an "info:" line
                lines.append(name + " " + " ".join(row))
                row = []
            continue
        code = int(fmt.encode(np.float64(x), saturate=True))
        row.append(f"{bits:08x}:{code:02x}")
        if len(row) == 8:
            lines.append(name + " " + " ".join(row))
            row = []
    if row:
        lines.append(name + " " + " ".join(row))
    return lines


def decoding(name, fmt, bfloat=False):
    lines = []
    for code in range(fmt.codes):
        value = float(fmt.decode(code))
        if value != value:
            lines.append(f"{name} code {code:02x} -> nan")
        elif bfloat:
            bits = int(np.array([value], dtype=np.float32).view(np.uint32)[0]) >> 16
            lines.append(f"{name} code {code:02x} -> bfloat16 {bits:04x}")
        else:
            bits = int(np.array([value], dtype=np.float16).view(np.uint16)[0])
            lines.append(f"{name} code {code:02x} -> half {bits:04x}")
    return lines


def predict(which: str) -> str:
    if which == "fp8":
        lines = (decoding("e4m3", E4M3) + decoding("e5m2", E5M2)
                 + encoding("e4m3", E4M3, sweep_inputs()) + encoding("e5m2", E5M2, sweep_inputs()))
    elif which == "fp4":
        lines = decoding("e2m1", E2M1) + encoding("e2m1", E2M1, sweep_inputs())
    elif which == "e8m0":
        lines = decoding("ue8m0", UE8M0, bfloat=True) + encoding("ue8m0", UE8M0, e8m0_inputs())
    elif which == "special":
        lines = [line for name, fmt in (("e4m3", E4M3), ("e5m2", E5M2), ("e2m1", E2M1), ("ue8m0", UE8M0))
                 for line in encoding(name, fmt, special_inputs())]
    else:
        raise SystemExit(__doc__)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(predict(sys.argv[1] if len(sys.argv) > 1 else ""))
