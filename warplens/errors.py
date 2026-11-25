"""Exception types. Every error names the PTX line it came from when one exists."""

from __future__ import annotations


class WarplensError(Exception):
    """Base class for all warplens errors."""


class PtxSyntaxError(WarplensError):
    """The PTX text could not be parsed."""


class UnsupportedError(WarplensError):
    """Valid PTX that warplens does not implement yet. Never silently skipped."""


# What the CUDA runtime returns for a launch it refuses before the kernel runs. Measured on an
# RTX 3080 Ti with CUDA 12.4 except where noted (validation/README.md, "Launch errors").
CUDA_INVALID_VALUE = 1  # cudaErrorInvalidValue: too much shared memory, a bad attribute value
CUDA_INVALID_CONFIGURATION = 9  # cudaErrorInvalidConfiguration: block or grid dimensions
CUDA_INVALID_CLUSTER_SIZE = 912  # cudaErrorInvalidClusterSize: a cluster shape the device refuses
CUDA_LAUNCH_FAILURE = 719  # cudaErrorLaunchFailure: the kernel faulted while running
# cudaErrorTensorMemoryLeak: the kernel returned with tensor memory still allocated. Measured
# on a B200 (validation/blackwell/measured/b200/tmem/alloc_leak.txt).
CUDA_TENSOR_MEMORY_LEAK = 721


class LaunchError(WarplensError):
    """A kernel launch was set up incorrectly on the host side.

    `cuda_error` is the code the CUDA runtime gives a program for this refusal; the shim
    returns it from the launch call (warplens/host/shim_server.py)."""

    def __init__(self, message: str, cuda_error: int = CUDA_INVALID_VALUE):
        super().__init__(message)
        self.cuda_error = cuda_error


class ExecutionError(WarplensError):
    """A kernel did something invalid while running."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.where: str | None = None

    def __str__(self) -> str:
        return self.message if self.where is None else f"{self.message}\n{self.where}"


class DeadlockError(ExecutionError):
    """Every unfinished warp is blocked and nothing can release them.

    `findings` holds what the checkers had reported up to that point, including
    any mbarrier whose byte count did not balance. The deadlock is the outcome;
    a finding there is often the cause.
    """

    def __init__(self, message: str, findings=()):
        super().__init__(message)
        self.findings = list(findings)


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
