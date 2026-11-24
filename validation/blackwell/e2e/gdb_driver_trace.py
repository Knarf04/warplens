"""The gdb side of e2e/driver_call_trace.py: log every call into the CUDA driver library and
save the arguments of every kernel launch. Not run directly; gdb sources it:

    WARPLENS_TRACE_DIR=DIR gdb -q -batch -x gdb_driver_trace.py --args PROGRAM ARGS...

It needs no symbols and no root: the program is started by gdb, and a breakpoint is put on
the address of every function `libcuda.so` exports (names `cu[A-Z]...`) and, where the
program uses the shared CUDA runtime, of every `cuda[A-Z]...` export of `libcudart.so`.
A runtime that is linked into another library (the CuTe DSL's) has no exports to stop at;
its calls are seen as the driver calls they make.

At a launch (`cuLaunchKernel`, `cuLaunchKernelEx` and their `_ptsz` forms) the driver is
asked, with `cuFuncGetParamInfo`, for the offset and size of every kernel parameter, and
that many bytes are read from each entry of `kernelParams`. This is how the 128-byte tensor
maps a CuTe DSL kernel is handed are saved (matrix row RT-11): they are kernel parameters,
or parts of one.

Written to DIR:
    calls.txt       every call in order, repeats of one name collapsed ("name x12")
    summary.txt     each distinct name and how often, in order of first call
    launches.txt    per launch: function, kernel name, grid, block, shared memory, and for
                    every parameter its offset, size and bytes in hex
    launches.json   the same for programs
    launchNN.paramMM.bin   the raw bytes of each parameter of at least 64 bytes
    exit.txt        how the program ended
"""

import json
import os
import re

import gdb

OUT = os.environ.get("WARPLENS_TRACE_DIR", ".")
MAX_PARAM_BYTES = 1 << 20   # per parameter; a kernel parameter is at most a few kilobytes
MAX_PARAMS = 256
MAX_STOPS = 100000
LIBRARIES = {r"libcuda\.so": re.compile(r"^cu[A-Z]\w*$"), r"libcudart\.so": re.compile(r"^cuda[A-Z]\w*$")}
LAUNCHES = ("cuLaunchKernel", "cuLaunchKernelEx", "cuLaunchKernel_ptsz", "cuLaunchKernelEx_ptsz")

for setting in ("pagination off", "confirm off", "breakpoint pending on", "print thread-events off",
                # Reading the symbols of every library a Python program loads takes minutes
                # (measured: over three with the CuTe DSL's compiler libraries, ten seconds
                # without). Only the traced libraries' symbols are read, on load.
                "auto-solib-add off", "breakpoint always-inserted on"):
    gdb.execute("set " + setting)
try:
    gdb.execute("set debuginfod enabled off")
except gdb.error:
    pass  # a gdb without debuginfod

calls = []          # names, in order
addresses = {}      # name -> address, of everything a breakpoint was set on
traced = set()      # library files already handled
notes = []
asking = [False]    # true while this script itself calls into the driver
exit_status = ["unknown"]


class Call(gdb.Breakpoint):
    def __init__(self, name, address):
        super().__init__(f"*{address:#x}", internal=True)
        self.name = name

    def stop(self):
        if asking[0]:               # one of this script's own questions to the driver
            return False
        calls.append(self.name)
        return self.name in LAUNCHES   # only a launch stops the program; the rest is logged


def trace_new_libraries():
    """Read the symbols of the traced libraries that are loaded, and break on their exports."""
    for pattern, wanted in LIBRARIES.items():
        try:
            gdb.execute("sharedlibrary " + pattern, to_string=True)
        except gdb.error:
            continue
        for objfile in gdb.objfiles():
            name = objfile.filename or ""
            if name in traced or not re.search(pattern, name):
                continue
            traced.add(name)
            count = 0
            # "Non-debugging symbols:" lines are "0x<address>  <name>"; a stripped library
            # has exactly its exports there.
            listing = gdb.execute("info functions ^cu", to_string=True)
            for line in listing.splitlines():
                match = re.match(r"^(0x[0-9a-f]+)\s+(\w+)$", line.strip())
                if not match or not wanted.match(match.group(2)) or match.group(2) in addresses:
                    continue
                if gdb.solib_name(int(match.group(1), 16)) != name:
                    continue
                addresses[match.group(2)] = int(match.group(1), 16)
                Call(match.group(2), int(match.group(1), 16))
                count += 1
            notes.append(f"tracing {count} exports of {name}")


def register(name):
    return int(gdb.parse_and_eval(f"(unsigned long)${name}"))


def read(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


def read_u32s(address, count):
    data = read(address, 4 * count)
    return [int.from_bytes(data[4 * i:4 * i + 4], "little") for i in range(count)]


def read_u64(address):
    return int.from_bytes(read(address, 8), "little")


scratch = [0]


def scratch_page():
    """A page of the program's memory for the driver to write answers to. From mmap, which
    takes no lock that a stopped thread could be holding (malloc might)."""
    if not scratch[0]:
        gdb.execute(r"sharedlibrary libc\.so", to_string=True)
        scratch[0] = int(gdb.parse_and_eval(
            "(unsigned long)((void *(*)(void *, unsigned long, int, int, int, long))mmap)(0, 4096, 3, 0x22, -1, 0)"))
        if scratch[0] in (0, 2 ** 64 - 1):
            raise gdb.error("mmap in the traced program failed")
    return scratch[0]


def driver_call(name, signature, *arguments):
    """Call an export of libcuda in the stopped program; returns its CUresult."""
    text = ", ".join(f"{a:#x}" if isinstance(a, int) else str(a) for a in arguments)
    asking[0] = True
    try:
        return int(gdb.parse_and_eval(f"(({signature}){addresses[name]:#x})({text})"))
    finally:
        asking[0] = False


def describe_launch(name, index):
    """What the program hands the driver at this launch. Stopped at the function's first
    instruction, so the arguments are where the calling convention puts them."""
    launch = {"index": index, "call": name}
    sp = register("rsp")
    if name.startswith("cuLaunchKernelEx"):     # (config *, CUfunction, void **kernelParams, void **extra)
        function, params = register("rsi"), register("rdx")
        grid_block = read_u32s(register("rdi"), 7)
    else:                                        # (f, gx, gy, gz, bx, by, bz, smem, stream, kernelParams, extra)
        function, params = register("rdi"), read_u64(sp + 32)
        grid_block = [register("rsi") & 0xFFFFFFFF, register("rdx") & 0xFFFFFFFF, register("rcx") & 0xFFFFFFFF,
                      register("r8") & 0xFFFFFFFF, register("r9") & 0xFFFFFFFF,
                      read_u32s(sp + 8, 1)[0], read_u32s(sp + 16, 1)[0]]
    launch.update(grid=grid_block[0:3], block=grid_block[3:6], shared_memory_bytes=grid_block[6],
                  kernel_params_pointer=params, parameters=[])
    page = scratch_page()
    gdb.execute("set scheduler-locking on")      # only this thread runs during the calls below
    try:
        if "cuFuncGetName" in addresses:
            if driver_call("cuFuncGetName", "int (*)(const char **, void *)", page, function) == 0:
                launch["kernel"] = gdb.parse_and_eval(f"*(char **){page:#x}").string()
        if "cuFuncGetParamInfo" not in addresses:
            launch["note"] = "this driver has no cuFuncGetParamInfo (CUDA 12.4 or newer); parameters not read"
            return launch
        if not params:
            launch["note"] = "kernelParams is null (the launch uses `extra`); parameters not read"
            return launch
        for i in range(MAX_PARAMS):
            status = driver_call("cuFuncGetParamInfo", "int (*)(void *, unsigned long, unsigned long *, unsigned long *)",
                                 function, i, page, page + 8)
            if status != 0:
                if i == 0:
                    launch["note"] = f"cuFuncGetParamInfo returned {status} for parameter 0; parameters not read"
                break
            offset, size = read_u64(page), read_u64(page + 8)
            data = read(read_u64(params + 8 * i), min(size, MAX_PARAM_BYTES))
            launch["parameters"].append({"index": i, "offset": offset, "size": size, "hex": data.hex()})
            if size >= 64:
                with open(os.path.join(OUT, f"launch{index:02d}.param{i:02d}.bin"), "wb") as f:
                    f.write(data)
    finally:
        gdb.execute("set scheduler-locking off")
    return launch


def collapse(names):
    lines, previous, count = [], None, 0
    for name in names + [None]:
        if name == previous:
            count += 1
            continue
        if previous is not None:
            lines.append(previous if count == 1 else f"{previous} x{count}")
        previous, count = name, 1
    return lines


def write_results(launches):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "calls.txt"), "w") as f:
        f.write("".join(f"# {note}\n" for note in notes))
        f.write("\n".join(collapse(calls)) + "\n")
    counts = {}
    for name in calls:
        counts[name] = counts.get(name, 0) + 1
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("".join(f"{name}\t{count}\n" for name, count in counts.items()))
    with open(os.path.join(OUT, "launches.json"), "w") as f:
        json.dump(launches, f, indent=1)
    with open(os.path.join(OUT, "launches.txt"), "w") as f:
        for launch in launches:
            f.write(f"launch {launch['index']}: {launch['call']} kernel={launch.get('kernel', '?')} "
                    f"grid={launch.get('grid')} block={launch.get('block')} "
                    f"shared_memory_bytes={launch.get('shared_memory_bytes')}\n")
            if "note" in launch:
                f.write(f"  note: {launch['note']}\n")
            for p in launch.get("parameters", []):
                mark = "  <-- 128 bytes or more: may hold a tensor map" if p["size"] >= 128 else ""
                f.write(f"  param {p['index']:2} offset {p['offset']:5} size {p['size']:5}{mark}\n")
                for at in range(0, len(p["hex"]), 64):
                    f.write(f"    {at // 2:04x}: {p['hex'][at:at + 64]}\n")
        if not launches:
            f.write("no kernel launch reached the driver\n")
    with open(os.path.join(OUT, "exit.txt"), "w") as f:
        f.write(exit_status[0] + "\n")


def on_exit(event):
    exit_status[0] = f"exit {event.exit_code}" if hasattr(event, "exit_code") else "ended (no exit code: killed by a signal)"


def main():
    launches = []
    gdb.events.exited.connect(on_exit)
    for pattern in LIBRARIES:
        gdb.execute("catch load " + pattern, to_string=True)
    try:
        gdb.execute("run")
        for _ in range(MAX_STOPS):
            if not gdb.selected_inferior().pid:
                break
            trace_new_libraries()    # a stop at a library load, or anything else: cheap, idempotent
            pc = register("pc")
            name = next((n for n in LAUNCHES if addresses.get(n) == pc), None)
            if name:
                try:
                    launches.append(describe_launch(name, len(launches)))
                except (gdb.error, gdb.MemoryError) as error:
                    launches.append({"index": len(launches), "call": name, "note": f"could not be read: {error}"})
            gdb.execute("continue")
        else:
            notes.append(f"stopped tracing after {MAX_STOPS} stops")
    except gdb.error as error:
        notes.append(f"gdb: {error}")
    finally:
        write_results(launches)


main()
