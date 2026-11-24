"""In-memory form of a parsed PTX module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# PTX fundamental type -> NumPy dtype used when an instruction interprets a value.
NP_TYPES: dict[str, np.dtype] = {
    "b8": np.dtype(np.uint8), "b16": np.dtype(np.uint16),
    "b32": np.dtype(np.uint32), "b64": np.dtype(np.uint64),
    "u8": np.dtype(np.uint8), "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32), "u64": np.dtype(np.uint64),
    "s8": np.dtype(np.int8), "s16": np.dtype(np.int16),
    "s32": np.dtype(np.int32), "s64": np.dtype(np.int64),
    "f16": np.dtype(np.float16), "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "pred": np.dtype(np.bool_),
}

UINT_BY_SIZE = {1: np.dtype(np.uint8), 2: np.dtype(np.uint16),
                4: np.dtype(np.uint32), 8: np.dtype(np.uint64)}
INT_BY_SIZE = {1: np.dtype(np.int8), 2: np.dtype(np.int16),
               4: np.dtype(np.int32), 8: np.dtype(np.int64)}


def storage_dtype(ty: str) -> np.dtype:
    """Registers hold raw bits; the instruction type decides how to read them."""
    dt = NP_TYPES[ty]
    return dt if dt.kind == "b" else UINT_BY_SIZE[dt.itemsize]


@dataclass(frozen=True, slots=True)
class Reg:
    name: str  # including the leading %


@dataclass(frozen=True, slots=True)
class Special:
    name: str  # e.g. "tid.x", without the leading %


@dataclass(frozen=True, slots=True)
class Imm:
    kind: str  # "int", "f32" or "f64"
    value: int  # the integer, or the IEEE bit pattern for f32/f64


@dataclass(frozen=True, slots=True)
class Sym:
    name: str  # a label or a parameter name


@dataclass(frozen=True, slots=True)
class Mem:
    base: Reg | Sym | Imm
    offset: int = 0


@dataclass(frozen=True, slots=True)
class Vec:
    elems: tuple


@dataclass(frozen=True, slots=True)
class Pair:
    """A paired destination such as `%r8|%p1` (value | predicate)."""
    first: Reg
    second: Reg


@dataclass(frozen=True, slots=True)
class TensorRef:
    """A TMA tensor operand: `[tensor_map, {c0, c1, ...}]`."""
    base: Reg | Sym | Imm
    coords: tuple


Operand = Reg | Special | Imm | Sym | Mem | Vec | Pair | TensorRef


@dataclass(slots=True)
class SharedVar:
    """A `.shared` variable. `size` is None for `.extern .shared name[]`, whose
    size is the dynamic shared memory requested at launch."""
    name: str
    align: int
    size: int | None
    line: int = 0


@dataclass(slots=True)
class Instr:
    opcode: str  # "ld"
    mods: tuple[str, ...]  # ("global", "b32")
    operands: tuple[Operand, ...]
    pred: Reg | None
    pred_negated: bool
    line: int
    text: str

    @property
    def name(self) -> str:
        return ".".join((self.opcode, *self.mods))


@dataclass(slots=True)
class Param:
    name: str
    type: str
    is_ptr: bool = False
    size: int = 0  # bytes
    by_value: bool = False  # a struct passed by value (`.b8 name[24]`)
    align: int = 0  # declared alignment, 0 if none


@dataclass(slots=True)
class GlobalVar:
    """A `.global` or `.const` variable with its initial bytes."""
    name: str
    align: int
    data: bytes
    line: int = 0


@dataclass(slots=True)
class Function:
    name: str
    params: list[Param] = field(default_factory=list)
    rets: list[Param] = field(default_factory=list)  # return values of a `.func`
    regs: dict[str, str] = field(default_factory=dict)  # name -> PTX type
    instrs: list[Instr] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)  # name -> instruction index
    shared: list[SharedVar] = field(default_factory=list)
    locals: list[SharedVar] = field(default_factory=list)  # `.local` variables, one copy per thread
    call_params: dict[str, str] = field(default_factory=dict)  # `.param`s declared for a call
    line: int = 0


@dataclass(slots=True)
class Module:
    name: str
    version: str = ""
    target: str = ""
    address_size: int = 64
    functions: dict[str, Function] = field(default_factory=dict)  # the kernels (`.entry`)
    device_functions: dict[str, Function] = field(default_factory=dict)  # `.func` with a body
    shared: list[SharedVar] = field(default_factory=list)
    globals: list[GlobalVar] = field(default_factory=list)
    extern_funcs: set[str] = field(default_factory=set)
