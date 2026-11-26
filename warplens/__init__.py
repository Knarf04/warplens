"""warplens: run Blackwell GPU kernels (PTX) on the CPU and watch what they do."""

from .errors import (
    DeadlockError,
    ExecutionError,
    LaunchError,
    MemoryFault,
    PtxSyntaxError,
    UnsupportedError,
    WarplensError,
)
from .frontend.ptx_parser import parse_ptx, parse_ptx_file
from .host.device import Device, DeviceArray, TensorMap
from .host.launch import LaunchStats
from .host.program import ProgramLaunch, ProgramRun, run_program

__version__ = "1.0.0"

__all__ = [
    "DeadlockError",
    "Device",
    "DeviceArray",
    "ExecutionError",
    "LaunchError",
    "LaunchStats",
    "MemoryFault",
    "ProgramLaunch",
    "ProgramRun",
    "PtxSyntaxError",
    "TensorMap",
    "UnsupportedError",
    "WarplensError",
    "parse_ptx",
    "parse_ptx_file",
    "run_program",
]
