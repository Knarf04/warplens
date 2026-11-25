import shutil
from pathlib import Path

import numpy as np
import pytest

import warplens

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HEADER = ".version 9.5\n.target sm_100a\n.address_size 64\n"


def cuda_include(header: str = "cuda.h") -> Path | None:
    """The CUDA toolkit include directory that has `header`, or None. Looks next to the nvcc on
    PATH first (a conda env keeps its headers under targets/), then in /usr/local/cuda."""
    candidates = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit = Path(nvcc).resolve().parent.parent
        candidates += [toolkit / "targets/x86_64-linux/include", toolkit / "include"]
    candidates.append(Path("/usr/local/cuda/include"))
    return next((p for p in candidates if (p / header).exists()), None)


def kernel(body: str, params: str = "", name: str = "k") -> warplens.frontend.ir.Module:
    """Wrap a function body in a module with one entry called `name`."""
    return warplens.parse_ptx(f"{HEADER}.visible .entry {name}({params})\n{{\n{body}\n}}\n",
                              name="test.ptx")


def run_lanes(sim, instr: str, in_types, out_type: str, inputs):
    """Run one instruction per lane: out[i] = instr(in0[i], in1[i], ...).

    `instr` uses %a0, %a1, ... for the inputs and %d for the result, e.g.
    "div.s32 %d, %a0, %a1;". Returns the output array.
    """
    n = len(inputs[0])
    bits = lambda ty: 1 if ty == "pred" else int(ty[1:])
    params = ", ".join(f".param .u64 p{i}" for i in range(len(inputs) + 1))
    lines = [".reg .b64 %base, %addr, %off;", ".reg .b32 %t;",
             f".reg .b{bits(out_type)} %d;", "mov.u32 %t, %tid.x;"]
    for i, ty in enumerate(in_types):
        size = bits(ty) // 8
        lines += [f".reg .b{bits(ty)} %a{i};",
                  f"ld.param.u64 %base, [p{i}];",
                  f"mul.wide.u32 %off, %t, {size};",
                  "add.u64 %addr, %base, %off;",
                  f"ld.global.b{bits(ty)} %a{i}, [%addr];"]
    out_size = bits(out_type) // 8
    lines += [instr,
              f"ld.param.u64 %base, [p{len(inputs)}];",
              f"mul.wide.u32 %off, %t, {out_size};",
              "add.u64 %addr, %base, %off;",
              f"st.global.b{bits(out_type)} [%addr], %d;",
              "ret;"]
    # Declarations must come first; keep statement order otherwise.
    decls = [l for l in lines if l.startswith(".reg")]
    code = [l for l in lines if not l.startswith(".reg")]
    mod = kernel("\n".join(decls + code), params)
    arrays = [sim.from_numpy(np.asarray(x)) for x in inputs]
    out = sim.empty(n, out_type)
    sim.launch(mod, "k", grid=1, block=n, args=[*arrays, out])
    return out.numpy()


@pytest.fixture
def sim():
    return warplens.Device("B200")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: runs on this machine's own CUDA GPU; left out where there is none")
    config.addinivalue_line("markers", "rental_archive(variable): reads a file of a rental's archive that is too "
                            "large for the repository; left out where the variable naming it is not set")


def pytest_collection_modifyitems(config, items):
    """Tests marked `gpu` are deselected, not skipped, on a run that has no GPU in reach
    (CUDA_VISIBLE_DEVICES set to nothing, or no nvidia-smi): CI's full job is such a run, and
    there a skip is a failure (WARPLENS_NO_SKIPS). With a GPU they run, and skip with a reason
    for whatever else they lack.

    Tests marked `rental_archive("VARIABLE")` are deselected the same way where VARIABLE is not
    set: what they read exists only in an archive brought back from a rented machine, and no
    install step can supply it. Where it is set they run, and skip if it names the wrong thing."""
    import os

    def left_out(item) -> bool:
        archive = item.get_closest_marker("rental_archive")
        return bool(archive and not os.environ.get(archive.args[0])) or bool(no_gpu and item.get_closest_marker("gpu"))

    no_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not shutil.which("nvidia-smi")
    deselected = [item for item in items if left_out(item)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not left_out(item)]


def pytest_sessionfinish(session, exitstatus):
    """With WARPLENS_NO_SKIPS=1 a skipped test fails the run. CI's full job sets it, so a
    missing toolchain or package there is an error instead of silently reduced coverage."""
    import os
    if os.environ.get("WARPLENS_NO_SKIPS") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        reporter.write_line(f"WARPLENS_NO_SKIPS=1: {len(skipped)} test(s) were skipped; failing the run",
                            red=True)
        for report in skipped:
            reporter.write_line(f"  {report.nodeid}: {report.longrepr[2] if isinstance(report.longrepr, tuple) else ''}")
        session.exitstatus = 1
