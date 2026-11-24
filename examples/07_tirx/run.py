"""A TIRx kernel that runs unchanged on a GPU and in warplens.

    pip install apache-tvm-tirx cuda-bindings
    python examples/07_tirx/run.py                 # on the GPU in this machine
    warplens run python examples/07_tirx/run.py    # in the simulator, as a B200

TIRx (the language of the book *Modern GPU Programming for MLSys*) loads and launches its
kernels through the CUDA driver. Under `warplens run` those calls reach the simulator, which
reports a compute capability of 10.0, so TIRx compiles for sm_100a. The one thing the script
has to ask for is PTX: by default TIRx hands the driver machine code, which only a real GPU
can run. A GPU accepts PTX as well, as long as its driver is at least as new as the NVRTC
that wrote the PTX.
"""

import numpy as np
import tvm_tirx
from tvm_tirx.script import tirx as T

N, BLOCK = 256, 64
PTX = {"cuda": {"target_format": "ptx"}}


@T.function
def saxpy(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")):
    with T.device_entry(launch=T.cuda.LaunchConfig(grid=N // BLOCK, block=BLOCK), backend_config=PTX):
        i = T.cuda.block_idx("x") * BLOCK + T.cuda.thread_idx("x")
        y[i] = T.float32(2.5) * x[i] + y[i]
        if i == N - 1:  # TIRx makes printf a device function, called rather than inlined
            T.cuda.printf("the last thread is %d of block %d\n", T.cuda.thread_idx("x"), T.cuda.block_idx("x"))


executable = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": saxpy}), target="cuda")

device = tvm_tirx.cuda(0)
rng = np.random.default_rng(0)
a, b = rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32)
x, y = tvm_tirx.runtime.tensor(a, device=device), tvm_tirx.runtime.tensor(b, device=device)
executable.mod(x, y)

np.testing.assert_allclose(y.numpy(), np.float32(2.5) * a + b, rtol=1e-6)
print(f"saxpy over {N} elements on compute capability {device.compute_version}: matches NumPy")
