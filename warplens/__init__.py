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
from .host.device import Device, DeviceArray
from .host.launch import LaunchStats

__version__ = "0.0.1"

__all__ = [
    "DeadlockError",
    "Device",
    "DeviceArray",
    "ExecutionError",
    "LaunchError",
    "LaunchStats",
    "MemoryFault",
    "PtxSyntaxError",
    "UnsupportedError",
    "WarplensError",
    "parse_ptx",
    "parse_ptx_file",
]
