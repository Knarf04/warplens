"""Exception types. Every error names the PTX line it came from when one exists."""

from __future__ import annotations


class WarplensError(Exception):
    """Base class for all warplens errors."""


class PtxSyntaxError(WarplensError):
    """The PTX text could not be parsed."""


class UnsupportedError(WarplensError):
    """Valid PTX that warplens does not implement yet. Never silently skipped."""


class LaunchError(WarplensError):
    """A kernel launch was set up incorrectly on the host side."""


class ExecutionError(WarplensError):
    """A kernel did something invalid while running."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.where: str | None = None

    def __str__(self) -> str:
        return self.message if self.where is None else f"{self.message}\n{self.where}"


class MemoryFault(ExecutionError):
    """A load or store touched bytes outside every allocation.

    `which` indexes the faulting accesses within the batch that was attempted;
    the executor converts it to lane numbers in `lanes`.
    """

    def __init__(self, message: str, which, addresses):
        super().__init__(message)
        self.which = which
        self.addresses = addresses
        self.lanes: list[int] = []
