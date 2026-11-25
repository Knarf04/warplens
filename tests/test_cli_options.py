"""`warplens run`: the options that reach `Device.launch` (`--clc-limit`, `--active-sms`,
`--timing-config`), and a Python script given without `python` in front of it.

Three kinds of test, named for what they need:
  - the options alone: parsing, and what `run` hands to the server and to the program;
  - `shim`: the real command on a program that launches through the CUDA driver (a C
    compiler and the CUDA headers, as for any `warplens run`);
  - Triton: the real command on a Triton script.
The TIRx and CuTe DSL scripts are in test_tirx.py and test_cute_dsl.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from warplens import cli
from warplens.host import runtime, shim_server
from warplens.timing.config import GpuConfig

from tests.support import EXAMPLES, REPO as ROOT

B200 = ROOT / "warplens" / "configs" / "b200.yaml"

# A host program without a compiler: the CUDA driver API through ctypes. It launches the
# cluster launch control example in two-CTA clusters and prints which CTA computed each tile.
HOST = r'''
import ctypes, sys
import numpy as np

TILE, TILES = 64, 12
cuda = ctypes.CDLL("libcuda.so.1")  # the names are those the library exports (cuda.h maps to _v2)

class Attribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("cluster", ctypes.c_uint * 3),
                ("rest", ctypes.c_char * 52)]

class Config(ctypes.Structure):
    _fields_ = [("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3), ("smem", ctypes.c_uint),
                ("stream", ctypes.c_void_p), ("attrs", ctypes.POINTER(Attribute)), ("count", ctypes.c_uint)]

def ok(status):
    if status:
        sys.exit(f"CUDA driver error {status}")

context, module, function = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
ok(cuda.cuInit(0))
ok(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), 0))
ok(cuda.cuCtxSetCurrent(context))
ok(cuda.cuModuleLoadData(ctypes.byref(module), open(sys.argv[1], "rb").read() + b"\0"))
ok(cuda.cuModuleGetFunction(ctypes.byref(function), module, b"scale_tiles"))
data = np.random.default_rng(0).standard_normal(TILES * TILE).astype(np.float32)
arrays = [ctypes.c_uint64() for _ in range(3)]  # in, out, worker
for array, size in zip(arrays, (data.nbytes, data.nbytes, 4 * TILES)):
    ok(cuda.cuMemAlloc_v2(ctypes.byref(array), ctypes.c_size_t(size)))
ok(cuda.cuMemcpyHtoD_v2(arrays[0], data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(data.nbytes)))
factor = ctypes.c_float(3.0)
params = (ctypes.c_void_p * 4)(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in (*arrays, factor)))
attribute = Attribute(id=4, cluster=(2, 1, 1))  # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
config = Config(grid=(TILES, 1, 1), block=(TILE, 1, 1), attrs=ctypes.pointer(attribute), count=1)
ok(cuda.cuLaunchKernelEx(ctypes.byref(config), function, params, None))
out, worker = np.empty_like(data), np.empty(TILES, np.uint32)
ok(cuda.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), arrays[1], ctypes.c_size_t(out.nbytes)))
ok(cuda.cuMemcpyDtoH_v2(worker.ctypes.data_as(ctypes.c_void_p), arrays[2], ctypes.c_size_t(worker.nbytes)))
print("output", "correct" if np.array_equal(out, np.float32(3.0) * data) else "WRONG")
print("workers", worker.tolist())
'''

TRITON = '''
import numpy as np
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
             mask=mask)

gpu = triton.runtime.driver.active  # the script installs nothing: `warplens run` has
a = np.arange(100, dtype=np.float32)
out = gpu.empty(100, "float32")
add_kernel[(4,)](gpu.tensor(a), gpu.tensor(a), out, 100, BLOCK=32)
print("triton script:", "matches NumPy" if np.array_equal(out.numpy(), a + a) else "WRONG")
timing = gpu.last_stats.timing
print("timing:", timing and (timing.active_sms, timing.config.name, timing.config["clock_ghz"]))
'''


def command(*arguments, **env):
    return subprocess.run([sys.executable, "-m", "warplens.cli", "run", *map(str, arguments)],
                          capture_output=True, text=True, timeout=900, cwd=ROOT,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env})


@pytest.fixture(scope="module")
def shim():
    """`warplens run` can start a program here: the shim libraries exist or can be built."""
    try:
        return cli.shim_libraries()
    except SystemExit as failure:
        pytest.skip(f"the shim libraries cannot be built here: {str(failure)[:200]}")


@pytest.fixture
def empty_shims(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(cli, "shim_libraries", lambda: lib)
    monkeypatch.setattr(runtime, "cute_dsl_runtime", lambda shims: None)
    return lib


@pytest.fixture
def servers(monkeypatch):
    """Every ShimServer that `run` makes, so that a test can see what it was given."""
    made = []

    class Recorded(shim_server.ShimServer):
        def __init__(self, *arguments, **options):
            super().__init__(*arguments, **options)
            made.append(self)

    monkeypatch.setattr(shim_server, "ShimServer", Recorded)
    return made


# -- the options alone -------------------------------------------------------------------------

@pytest.mark.parametrize("arguments, message", [
    (["--clc-limit", "-1"], "argument --clc-limit: must be an integer that is at least 0, got '-1'"),
    (["--clc-limit", "many"], "argument --clc-limit: must be an integer that is at least 0, got 'many'"),
    (["--clc-limit", "1.5"], "argument --clc-limit: must be an integer that is at least 0, got '1.5'"),
    (["--active-sms", "0"], "argument --active-sms: must be an integer that is at least 1, got '0'"),
    (["--active-sms", "149"], "argument --active-sms: must be from 1 to 148, the SM count of timing config `B200`"),
    (["--timing-config", "no/such/config.yaml"], "argument --timing-config: no timing config at no/such/config.yaml"),
    (["--timing-config", "b100"], "argument --timing-config: no timing config at "),
])
def test_a_bad_value_is_refused_before_anything_runs(arguments, message, capsys, monkeypatch):
    monkeypatch.setattr(cli, "run", lambda args: pytest.fail("the program was started"))
    with pytest.raises(SystemExit) as exit:
        cli.main(["run", *arguments, sys.executable, "-c", "pass"])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert message in error and "Traceback" not in error


def test_a_timing_config_that_cannot_be_used_is_refused_with_its_reason(tmp_path, capsys):
    broken = tmp_path / "broken.yaml"
    broken.write_text("sm_count: {value: 148}\n")  # no source
    with pytest.raises(SystemExit):
        cli.main(["run", "--timing-config", str(broken), sys.executable, "-c", "pass"])
    assert "argument --timing-config: broken.yaml: entry `sm_count` needs a `value` and a `source`" \
        in capsys.readouterr().err
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("sm_count: {value: 8, source: a test}\n")
    with pytest.raises(SystemExit):
        cli.main(["run", "--timing-config", str(incomplete), sys.executable, "-c", "pass"])
    assert "argument --timing-config: timing config `incomplete` is missing entries" in capsys.readouterr().err


def test_active_sms_is_checked_against_the_config_that_is_used(tmp_path, capsys):
    small = tmp_path / "small.yaml"
    small.write_text(B200.read_text().replace("value: 148", "value: 8", 1))
    assert GpuConfig.load(small)["sm_count"] == 8
    with pytest.raises(SystemExit):
        cli.main(["run", "--timing-config", str(small), "--active-sms", "9", sys.executable, "-c", "pass"])
    assert "argument --active-sms: must be from 1 to 8" in capsys.readouterr().err


def test_help_describes_the_options(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    for expected in ("--clc-limit N", "--active-sms N", "--timing-config CONFIG", "cluster launch control",
                     "Implies --timing", "a Python script (a .py file)"):
        assert expected in text


def test_options_reach_the_server_and_the_program(empty_shims, servers, tmp_path, capfd):
    small = tmp_path / "small.yaml"
    small.write_text(B200.read_text().replace("value: 148", "value: 8", 1))
    code = cli.main(["run", "--quiet", "--model", "B300", "--timing-config", str(small), "--active-sms", "4",
                     "--clc-limit", "3", sys.executable, "-c",
                     "import os; print(os.environ['WARPLENS_RUN_OPTIONS'])"])
    assert code == 0
    (server,) = servers
    assert isinstance(server.timing, GpuConfig) and server.timing["sm_count"] == 8
    assert (server.active_sms, server.clc_limit, server.device.model) == (4, 3, "B300")
    told = json.loads(capfd.readouterr().out)
    report = told.pop("report")
    assert not Path(report).exists()  # removed once the program has ended
    package = Path(cli.__file__).resolve().parent.parent  # the checkout, or site-packages when installed
    assert told == {"model": "B300", "timing": False, "timing_config": str(small), "active_sms": 4,
                    "clc_limit": 3, "package": str(package)}


def test_without_the_options_a_launch_is_what_it_was(empty_shims, servers):
    assert cli.main(["run", "--quiet", sys.executable, "-c", "pass"]) == 0
    assert (servers[0].timing, servers[0].active_sms, servers[0].clc_limit) == (False, None, None)
    assert cli.main(["run", "--quiet", "--timing", sys.executable, "-c", "pass"]) == 0
    assert (servers[1].timing, servers[1].active_sms, servers[1].clc_limit) == (True, None, None)
    assert cli.main(["run", "--quiet", "--active-sms", "2", sys.executable, "-c", "pass"]) == 0
    assert (servers[2].timing, servers[2].active_sms) == (True, 2)  # the SM count is for the timing model


def test_the_server_passes_its_options_to_every_launch():
    """From a launch request to the kernel of examples/09_clc, without a program or the shim."""
    import struct
    from tests.support.shim import Client, fatbin
    stats = {}
    for limit in (None, 0):
        c = Client(clc_limit=limit, timing=True, active_sms=4)
        (module,) = struct.unpack("<I", c.ok(shim_server.REGISTER_FATBIN,
                                             fatbin((EXAMPLES / "09_clc" / "clc.ptx").read_text())))
        function = struct.unpack_from("<I", c.ok(shim_server.REGISTER_FUNCTION,
                                                 struct.pack("<I", module) + b"scale_tiles"))[0]
        arrays = [struct.unpack("<Q", c.ok(shim_server.MALLOC, struct.pack("<Q", size)))[0]
                  for size in (3072, 3072, 48)]
        c.ok(shim_server.WRITE, struct.pack("<Q", arrays[0]) + bytes(3072))
        c.ok(shim_server.LAUNCH, struct.pack("<10IQ", function, 12, 1, 1, 64, 1, 1, 2, 1, 1, 0)
             + struct.pack("<3Qf", *arrays, 3.0))
        ((_, stats[limit]),) = c.server.launches
        assert stats[limit].timing is not None and stats[limit].timing.active_sms == 4
    assert (stats[None].canceled_ctas, stats[0].canceled_ctas) == (10, 0)


# -- a Python script is run by this interpreter, under the shim --------------------------------

def test_a_script_is_started_like_a_program_given_after_python(empty_shims, servers, tmp_path, capfd):
    script = tmp_path / "script.py"
    script.write_text("import os, sys\n"
                      "print(os.getpid(), sys.argv[1:], bool(os.environ.get('WARPLENS_SHIM_FD')),\n"
                      "      os.environ['WARPLENS_MODEL'], 'triton' in sys.modules)\n"
                      "raise SystemExit(7)\n")
    assert cli.main(["run", "--model", "B300", str(script), "--flag", "value"]) == 7
    captured = capfd.readouterr()
    pid, rest = captured.out.split(maxsplit=1)
    assert int(pid) != os.getpid()
    # Triton is not imported for a script that does not import it.
    assert rest.strip() == "['--flag', 'value'] True B300 False"
    # The empty shim directory makes the loader complain first; then no advice about nvcc.
    assert captured.err.strip().splitlines()[-1] == "warplens: the script launched no kernels"
    assert "nvcc" not in captured.err


def test_a_script_falls_back_to_this_process_when_there_is_no_shim(monkeypatch, tmp_path, capsys):
    pytest.importorskip("triton")

    def no_shim():
        raise SystemExit("warplens: could not build the shim libraries: no compiler")

    monkeypatch.setattr(cli, "shim_libraries", no_shim)
    script = tmp_path / "script.py"
    script.write_text("import os\nraise SystemExit(os.getpid())\n")
    assert cli.main(["run", "--quiet", str(script)]) == os.getpid()
    error = capsys.readouterr().err
    assert "could not build the shim libraries: no compiler" in error
    assert "without the shim libraries: its Triton kernels are simulated" in error
    with pytest.raises(SystemExit, match="could not build the shim libraries"):  # a program has no such way out
        cli.main(["run", "--quiet", sys.executable, str(script)])


def test_the_start_up_hook_installs_the_triton_driver_when_triton_is_imported(tmp_path):
    """With a stand-in for Triton and for warplens: what the hook does, and when."""
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "__init__.py").write_text("print('triton imported')\n")
    (tmp_path / "package" / "warplens").mkdir(parents=True)
    (tmp_path / "package" / "warplens" / "__init__.py").write_text("")
    (tmp_path / "package" / "warplens" / "cli.py").write_text(
        "def install_triton_driver():\n    import sys\n    print('driver installed', 'triton' in sys.modules)\n")
    hook = Path(cli.__file__).resolve().parent / "shim" / "python"
    # -S, and the hook imported by hand as `site` would: without site-packages, a warplens that
    # is installed there (not editable) cannot be found before the stand-in, which the hook
    # puts last on the path.
    code = "import sitecustomize, sys\nprint('started')\nimport triton\nprint(sys.path.count(sys.argv[1]))"

    def child(**env):
        env = {**{k: v for k, v in os.environ.items() if not k.startswith("WARPLENS_")},
               "PYTHONPATH": f"{hook}{os.pathsep}{tmp_path}", **env}
        done = subprocess.run([sys.executable, "-S", "-c", code, str(tmp_path / "package")], capture_output=True,
                              text=True, timeout=120, env=env, cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        return done

    options = json.dumps({"package": str(tmp_path / "package")})
    hooked = child(WARPLENS_SHIM_FD="99", WARPLENS_RUN_OPTIONS=options)
    assert hooked.stdout.splitlines() == ["started", "triton imported", "driver installed True", "0"]
    assert child().stdout.splitlines() == ["started", "triton imported", "0"]  # not under `warplens run`
    failed = child(WARPLENS_SHIM_FD="99", WARPLENS_RUN_OPTIONS=json.dumps({"package": str(tmp_path / "nowhere")}))
    if "driver installed" not in failed.stdout:  # unless a warplens is installed for this interpreter
        assert "could not install the Triton driver" in failed.stderr and "triton imported" in failed.stdout


# -- the real command --------------------------------------------------------------------------

@pytest.mark.parametrize("arguments, workers, summary", [
    ([], [0, 1] * 6, "scale_tiles: 12 CTA(s) of which 10 canceled before starting, "),
    (["--clc-limit", "2"], [0, 1] * 3 + [6, 7] * 3, "scale_tiles: 12 CTA(s) of which 8 canceled before starting, "),
    (["--clc-limit", "0"], list(range(12)), "scale_tiles: 12 CTA(s), "),
])
def test_clc_limit_decides_how_many_clusters_of_a_program_start(shim, tmp_path, arguments, workers, summary):
    host = tmp_path / "host.py"
    host.write_text(HOST)
    done = command(*arguments, host, EXAMPLES / "09_clc" / "clc.ptx")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert done.stdout.splitlines() == ["output correct", f"workers {workers}"]
    assert "1 kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr and summary in done.stderr


def slow_clock(tmp_path) -> Path:
    """The B200 timing config with a clock of 0.5 GHz: everything counted in cycles takes longer."""
    lines = B200.read_text().splitlines()
    at = lines.index("clock_ghz:")
    value = next(i for i in range(at, len(lines)) if "value:" in lines[i])
    lines[value] = lines[value].split("value:")[0] + "value: 0.5"
    slow = tmp_path / "slow.yaml"
    slow.write_text("\n".join(lines) + "\n")
    assert GpuConfig.load(slow)["clock_ghz"] == 0.5 != GpuConfig.load("B200")["clock_ghz"]
    return slow


def test_the_timing_config_decides_the_simulated_time_of_a_program(shim, tmp_path):
    host = tmp_path / "host.py"
    host.write_text(HOST)
    slow = slow_clock(tmp_path)

    def simulated(*arguments):
        done = command(*arguments, sys.executable, host, EXAMPLES / "09_clc" / "clc.ptx")
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
        (line,) = [line for line in done.stderr.splitlines() if "scale_tiles" in line]
        return float(line.split(" ns simulated")[0].rsplit(", ", 1)[1].replace(",", "")) if "simulated" in line else None

    assert simulated() is None
    plain = simulated("--timing")
    assert plain is not None
    assert simulated("--timing-config", "b200") == plain  # the built-in config, by name; implies --timing
    assert simulated("--timing-config", slow) > plain
    assert simulated("--active-sms", "4") == plain  # implies --timing; this kernel is not bound by memory


def test_a_triton_script_gets_the_options_in_its_own_process(shim, tmp_path):
    """Triton launches in the script's process, so the options have to arrive there."""
    pytest.importorskip("triton")
    script = tmp_path / "script.py"
    script.write_text(TRITON)
    slow = slow_clock(tmp_path)
    done = command(script)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert done.stdout.splitlines() == ["triton script: matches NumPy", "timing: None"]
    assert "1 kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr
    assert "add_kernel: 4 CTA(s)" in done.stderr and "simulated (" not in done.stderr
    clock = {name: GpuConfig.load(name)["clock_ghz"] for name in ("B200", "B300")}
    for arguments, timing in ((["--timing"], (148, "B200", clock["B200"])),
                              (["--active-sms", "4"], (4, "B200", clock["B200"])),
                              (["--timing-config", slow, "--active-sms", "5"], (5, "B200", 0.5)),
                              (["--model", "B300", "--timing"], (int(GpuConfig.load("B300")["sm_count"]), "B300", clock["B300"]))):
        timed = command(*arguments, script)
        assert timed.returncode == 0, timed.stdout[-1500:] + timed.stderr[-1500:]
        assert timed.stdout.splitlines()[1] == f"timing: {timing}"
        assert "add_kernel: 4 CTA(s)" in timed.stderr and " ns simulated (" in timed.stderr
    quiet = command("--quiet", script)
    assert quiet.returncode == 0 and "kernel launch" not in quiet.stderr


def test_a_triton_script_also_runs_when_python_is_spelled_out(shim, tmp_path):
    pytest.importorskip("triton")
    script = tmp_path / "script.py"
    script.write_text(TRITON)
    done = command(sys.executable, script)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "triton script: matches NumPy" in done.stdout and "add_kernel: 4 CTA(s)" in done.stderr
