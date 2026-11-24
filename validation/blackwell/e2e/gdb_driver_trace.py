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
that many bytes are read from each entry of `kernelParams`. (A launch that is given a
CUkernel in place of a CUfunction, as the CUDA runtime's own launches are, PyTorch's among
them, gets error 400 there and its parameters are not read; when capturing, see below,
`cuKernelGetParamInfo` is asked instead.) This is how the 128-byte tensor
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

More is saved when WARPLENS_TRACE_CAPTURE asks for it (comma-separated; e2e/cute_dsl_tma.py
sets both, e2e/launch_capture.py checks the result):

  module   The device code the program hands the driver. At cuModuleLoadData[Ex],
           cuModuleLoadFatBinary, cuLibraryLoadData and cuLinkAddData the image is saved as
           moduleNN.bin (the driver is given no size: PTX text ends at its NUL, an ELF and a
           fat binary say their size in their headers); at cuModuleLoad and
           cuLibraryLoadFromFile the named file is copied. At cuModuleGetFunction and
           cuLibraryGetKernel the name asked for is recorded. Each call's result and the
           handle it returned are read where the call returns. All in capture.json.
  memory   The device memory a launch can reach, before the kernel runs and after it has
           finished. Allocations are followed (cuMemAlloc and its variants, cuMemFree), read
           where each call returns. At a launch every 8 bytes of every parameter, at every
           byte offset, are held against the live allocations; a value inside none of them is
           put to the driver (cuMemGetAddressRange), which knows allocations made out of sight.
           Each allocation a parameter points into is read out whole with cuMemcpyDtoH, called
           in the stopped program; so is every other live allocation when the launch has a
           parameter of 64 bytes or more (a tensor map may hold its address in a form the scan
           does not see). Where the launch call returns, cuCtxSynchronize is called and the same
           allocations are read again. Files are mem/<sha256>.bin, so memory that did not
           change is stored once; launches.json gets a "memory" entry per launch naming, for
           each pointer, the parameter, the offset, the allocation and its two dumps.
           A PyTorch tensor lives inside a larger allocation of PyTorch's caching allocator;
           the dump is that whole allocation, and the pointer's offset in it is recorded.
           WARPLENS_TRACE_MODULE_SKIP_WRAPPED=1 leaves out the modules a library registers
           through nvcc's wrapper (PyTorch's and cuBLAS's own kernels, a megabyte each): size
           and SHA-256 are recorded, the bytes are not kept.
           WARPLENS_TRACE_MEMORY_SKIP is a regular expression: a launch whose kernel name it
           matches has its parameters saved as always and no memory read (cute_dsl_tma.py
           gives `^_Z`: PyTorch's own kernels, whose names are C++ ones).
           Threads. The calls this script makes for `memory`, and cuKernelGetParamInfo, run
           with the program's other threads running too. With only the stopped thread
           running (gdb's scheduler-locking, which the questions about a CUfunction have
           always used), cuKernelGetParamInfo at a launch by PyTorch never returned on
           driver 580.173.02: it waits for another thread. WARPLENS_TRACE_LOCK_THREADS=1
           runs them with the other threads held all the same. A driver call another thread
           makes during one of these calls is not followed; capture.json counts them
           ("unseen_during_calls"). gdb 15 and newer gives up a call that has not returned
           after WARPLENS_TRACE_CALL_SECONDS (default 60) and records it; capture.log has a
           line per call, written before it is made.
           Limits: WARPLENS_TRACE_MAX_ALLOCATION_BYTES per allocation (default 256 MiB),
           WARPLENS_TRACE_MAX_TOTAL_BYTES read in all (default 2 GiB), and for modules
           WARPLENS_TRACE_MAX_MODULE_BYTES saved in all (default 512 MiB). What is not read
           is recorded with the reason, never left out silently.
"""

import hashlib
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

CAPTURE = {word.strip() for word in os.environ.get("WARPLENS_TRACE_CAPTURE", "").split(",") if word.strip()}
MAX_ALLOCATION_BYTES = int(os.environ.get("WARPLENS_TRACE_MAX_ALLOCATION_BYTES", 256 << 20))
MAX_TOTAL_BYTES = int(os.environ.get("WARPLENS_TRACE_MAX_TOTAL_BYTES", 2 << 30))
MAX_MODULE_BYTES = int(os.environ.get("WARPLENS_TRACE_MAX_MODULE_BYTES", 512 << 20))   # all modules together
MEMORY_SKIP = os.environ.get("WARPLENS_TRACE_MEMORY_SKIP", "")
SKIP_WRAPPED = os.environ.get("WARPLENS_TRACE_MODULE_SKIP_WRAPPED", "") == "1"
LOCK_THREADS = "on" if os.environ.get("WARPLENS_TRACE_LOCK_THREADS", "") == "1" else "off"
MAX_RANGE_QUERIES = 4096    # questions to the driver per launch about values no allocation explains
CHUNK = 4 << 20             # device memory is read through a host buffer of this size
# Calls with (result *, image, ...): the image is the second argument and has no size.
MODULE_IMAGES = ("cuModuleLoadData", "cuModuleLoadDataEx", "cuModuleLoadFatBinary", "cuLibraryLoadData",
                 "cudaLibraryLoadData")
MODULE_FILES = ("cuModuleLoad", "cuLibraryLoadFromFile", "cudaLibraryLoadFromFile")      # (result *, file name, ...)
LINK_DATA = ("cuLinkAddData", "cuLinkAddData_v2")       # (state, type, data, size, name, ...)
LOOKUPS = ("cuModuleGetFunction", "cuLibraryGetKernel", "cudaLibraryGetKernel")           # (result *, handle, name)
HANDLE_OF = ("cuKernelGetFunction", "cuLibraryGetModule")   # (result *, handle): one handle made from another
ALLOCATIONS = ("cuMemAlloc", "cuMemAlloc_v2", "cuMemAllocManaged", "cuMemAllocAsync", "cuMemAllocFromPoolAsync")
PITCHED = ("cuMemAllocPitch", "cuMemAllocPitch_v2")     # (dptr *, pitch *, width bytes, height, element bytes)
FREES = ("cuMemFree", "cuMemFree_v2", "cuMemFreeAsync")
STOPS = set(LAUNCHES)
if "module" in CAPTURE:
    STOPS.update(MODULE_IMAGES + MODULE_FILES + LINK_DATA + LOOKUPS + HANDLE_OF)
if "memory" in CAPTURE:
    STOPS.update(ALLOCATIONS + PITCHED + FREES)

for setting in ("pagination off", "confirm off", "breakpoint pending on", "print thread-events off",
                # Reading the symbols of every library a Python program loads takes minutes
                # (measured: over three with the CuTe DSL's compiler libraries, ten seconds
                # without). Only the traced libraries' symbols are read, on load.
                "auto-solib-add off", "breakpoint always-inserted on"):
    gdb.execute("set " + setting)
# A call this script makes in the program that faults is undone, not left as the program's
# state. The setting has had two names.
for setting in ("unwind-on-signal on", "unwindonsignal on"):
    try:
        gdb.execute("set " + setting, to_string=True)
        break
    except gdb.error:
        pass
try:
    gdb.execute("set debuginfod enabled off")
except gdb.error:
    pass  # a gdb without debuginfod
# A call this script makes in the program that does not return (a lock held by a thread that
# is not running) is given up after CALL_SECONDS and undone. gdb 15 and newer; an older gdb
# waits until driver_call_trace.py's own timeout, and keeps what was already written.
CALL_SECONDS = int(os.environ.get("WARPLENS_TRACE_CALL_SECONDS", 60))
call_timeouts = True
for setting in (f"direct-call-timeout {CALL_SECONDS}", "unwind-on-timeout on"):
    try:
        gdb.execute("set " + setting)
    except gdb.error:
        call_timeouts = False

calls = []          # names, in order
addresses = {}      # name -> address, of everything a breakpoint was set on
traced = set()      # library files already handled
notes = []
asking = [False]    # true while this script itself calls into the driver
asking_thread = [0]
unseen = []         # captured calls other threads made meanwhile
exit_status = ["unknown"]


class Call(gdb.Breakpoint):
    def __init__(self, name, address):
        super().__init__(f"*{address:#x}", internal=True)
        self.name = name

    def stop(self):
        if asking[0]:               # one of this script's own questions to the driver
            if self.name in STOPS and gdb.selected_thread().num != asking_thread[0]:
                unseen.append(self.name)    # another thread's call, while it could not be stopped at
            return False
        calls.append(self.name)
        return self.name in STOPS   # a launch stops the program, and what is captured; the rest is logged


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


pending = {}        # (thread, stack pointer after the return) -> (return address, what to do there)
return_points = {}  # address -> Return, for as long as a call is to return there


class Return(gdb.Breakpoint):
    """Where a captured call returns to. It stops only the return that was asked for: the
    same address is passed by other threads and by calls this script makes."""

    def __init__(self, address):
        super().__init__(f"*{address:#x}", internal=True)

    def stop(self):
        return not asking[0] and (gdb.selected_thread().num, register("rsp")) in pending


def at_return(record):
    """Stopped at the first instruction of a call: come back when it returns."""
    sp = register("rsp")
    address = read_u64(sp)
    if address not in return_points:
        return_points[address] = Return(address)
    pending[(gdb.selected_thread().num, sp + 8)] = (address, record)


def returned(key):
    """The call has returned: its record, and its return point removed unless another call
    is still to return there. A return point left behind in a library that is unloaded
    later (the CuTe DSL's runtime, at exit) is a breakpoint gdb 18 cannot insert again,
    and the program then does not go on."""
    address, record = pending.pop(key)
    if not any(other == address for other, _ in pending.values()):
        return_points.pop(address).delete()
    return record


def go_on():
    try:
        gdb.execute("continue")
    except gdb.error as error:
        if "Cannot insert breakpoint" not in str(error) or not return_points:
            raise
        # A call that never returned left its return point in a library that is gone.
        notes.append("gdb: " + " ".join(str(error).split()) + "; the return points were removed")
        for point in return_points.values():
            point.delete()
        return_points.clear()
        pending.clear()
        gdb.execute("continue")


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


def progress(text):
    """One line to DIR/capture.log, written at once: if the tracer is killed at a call that
    never returns, the last line names it."""
    if CAPTURE:
        try:
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, "capture.log"), "a") as f:
                f.write(text + "\n")
        except OSError:
            pass    # the log is an aid; the capture does not depend on it


def driver_call(name, signature, *arguments):
    """Call an export of libcuda in the stopped program; returns its CUresult."""
    text = ", ".join(f"{a:#x}" if isinstance(a, int) else str(a) for a in arguments)
    progress(f"call {name}({text})")
    asking[0], asking_thread[0] = True, gdb.selected_thread().num
    try:
        result = int(gdb.parse_and_eval(f"(({signature}){addresses[name]:#x})({text})"))
    finally:
        asking[0] = False
    progress(f"  = {result}")
    return result


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
        if "kernel" not in launch:      # a CUkernel: its name is the one its lookup asked for
            name = next((l["name"] for l in capture["lookups"] if l.get("handle") == function and "name" in l), None)
            if name:
                launch["kernel"] = name
        if "cuFuncGetParamInfo" not in addresses:
            launch["note"] = "this driver has no cuFuncGetParamInfo (CUDA 12.4 or newer); parameters not read"
            return launch
        if not params:
            launch["note"] = "kernelParams is null (the launch uses `extra`); parameters not read"
            return launch
        ask, signature = "cuFuncGetParamInfo", "int (*)(void *, unsigned long, unsigned long *, unsigned long *)"
        for i in range(MAX_PARAMS):
            status = driver_call(ask, signature, function, i, page, page + 8)
            if status != 0 and i == 0 and CAPTURE and "cuKernelGetParamInfo" in addresses \
                    and not (MEMORY_SKIP and re.search(MEMORY_SKIP, launch.get("kernel", ""))):
                # Not a CUfunction; a CUkernel, then. Not with the other threads held (the header).
                gdb.execute("set scheduler-locking " + LOCK_THREADS)
                ask = launch["parameters_from"] = "cuKernelGetParamInfo"
                status = driver_call(ask, signature, function, i, page, page + 8)
            if status != 0:
                if i == 0:
                    launch["note"] = f"{ask} returned {status} for parameter 0; parameters not read"
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


# ---- WARPLENS_TRACE_CAPTURE: the device code, and the memory a launch can reach ----

capture = {"capture": sorted(CAPTURE), "call_timeout_seconds": CALL_SECONDS if call_timeouts else None,
           "modules": [], "lookups": [], "handles": [],
           "allocations": [], "problems": []}
live = {}           # base -> size, of the device allocations the program holds now
read_total = [0]    # bytes of device memory read so far
module_total = [0]  # bytes of modules saved so far
copy_buffer = [0]
FAILURES = (gdb.error, gdb.MemoryError, OSError, ValueError, KeyError)


def guarded(function, argument):
    """A capture step that fails says so in capture.json; the program goes on."""
    try:
        function(argument)
    except FAILURES as error:
        capture["problems"].append(f"{function.__name__}: {error}")


def result_of_call():
    return int(gdb.parse_and_eval("(int)$eax"))


def read_text(address, limit):
    """Bytes up to a NUL, read a page at a time so that the end of a mapping is not crossed."""
    data = b""
    while len(data) < limit:
        piece = read(address + len(data), 4096 - (address + len(data)) % 4096)
        end = piece.find(b"\0")
        if end >= 0:
            return data + piece[:end]
        data += piece
    raise ValueError(f"no NUL within {limit} bytes of {address:#x}")


def image_extent(address):
    """(format, address, bytes) of a module image, which the driver is handed without a size."""
    head = read(address, 64)
    field = lambda data, at, width: int.from_bytes(data[at:at + width], "little")
    if head[:4] == bytes.fromhex("b1436246"):                    # what nvcc links in: {magic, version, fat binary *}
        return ("wrapped " + (inner := image_extent(field(head, 8, 8)))[0], *inner[1:])
    if head[:4] == b"\x7fELF" and head[4] == 2:
        section_table, entry, count = field(head, 0x28, 8), field(head, 0x3A, 2), field(head, 0x3C, 2)
        size = max(64, section_table + entry * count, field(head, 0x20, 8) + field(head, 0x36, 2) * field(head, 0x38, 2))
        for i in range(count):
            section = read(address + section_table + entry * i, 64)
            if field(section, 4, 4) != 8:                        # SHT_NOBITS takes no room in the file
                size = max(size, field(section, 0x18, 8) + field(section, 0x20, 8))
        return "elf", address, size
    if head[:4] == bytes.fromhex("50ed55ba"):                    # a fat binary: 16-byte header, then its payload
        return "fatbin", address, field(head, 6, 2) + field(head, 8, 8)
    if not all(byte in b"\t\n\r" or 32 <= byte < 127 for byte in head.split(b"\0")[0]) or head[:1] == b"\0":
        raise ValueError(f"not an ELF, a fat binary or text; it starts with {head[:16].hex()}")
    return "text", address, len(read_text(address, MAX_MODULE_BYTES)) + 1   # PTX, with its NUL


def save_module(record, index, data):
    if module_total[0] + len(data) > MAX_MODULE_BYTES:
        raise ValueError(f"WARPLENS_TRACE_MAX_MODULE_BYTES ({MAX_MODULE_BYTES}) would be passed "
                         f"({module_total[0]} saved so far)")
    module_total[0] += len(data)
    name = f"module{index:02d}.bin"
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)
    record.update(file=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())


def returned_handle(record):
    record["result"] = result_of_call()
    pointer = record.pop("handle_pointer")
    record["handle"] = read_u64(pointer) if record["result"] == 0 else None


def module_image(name):
    record = {"call": name, "handle_pointer": register("rdi"), "then": returned_handle}
    capture["modules"].append(record)       # listed even when its bytes cannot be read
    at_return(record)
    try:
        record["format"], address, size = image_extent(register("rsi"))
        if size > MAX_MODULE_BYTES:
            raise ValueError(f"a {record['format']} image of {size} bytes is over the limit")
        if SKIP_WRAPPED and record["format"].startswith("wrapped"):
            record.update(size=size, sha256=hashlib.sha256(read(address, size)).hexdigest(),
                          not_kept="linked into a library (WARPLENS_TRACE_MODULE_SKIP_WRAPPED)")
            return
        save_module(record, len(capture["modules"]) - 1, read(address, size))
    except FAILURES as error:
        record["error"] = f"not saved: {error}"


def module_file(name):
    record = {"call": name, "handle_pointer": register("rdi"), "then": returned_handle, "format": "file"}
    capture["modules"].append(record)
    at_return(record)
    try:
        record["path"] = read_text(register("rsi"), 4096).decode(errors="replace")
        with open(record["path"], "rb") as f:
            save_module(record, len(capture["modules"]) - 1, f.read())
    except FAILURES as error:
        record["error"] = f"not saved: {error}"


def link_data(name):
    record = {"call": name, "format": "link input", "input_type": register("rsi") & 0xFFFFFFFF, "handle": None}
    capture["modules"].append(record)
    try:
        size = register("rcx")
        if size > MAX_MODULE_BYTES:
            raise ValueError(f"an input of {size} bytes is over the limit")
        save_module(record, len(capture["modules"]) - 1, read(register("rdx"), size))
    except FAILURES as error:
        record["error"] = f"not saved: {error}"


def lookup(name):
    record = {"call": name, "handle_pointer": register("rdi"), "module_handle": register("rsi"), "then": returned_handle}
    capture["lookups"].append(record)
    at_return(record)
    record["name"] = read_text(register("rdx"), 65536).decode(errors="replace")


def function_handle(name):
    record = {"call": name, "handle_pointer": register("rdi"), "from_handle": register("rsi"), "then": returned_handle}
    capture["handles"].append(record)
    at_return(record)


def allocated(record):
    record["result"] = result_of_call()
    base_pointer, pitch_pointer = record.pop("base_pointer"), record.pop("pitch_pointer", None)
    if record["result"] != 0:
        return
    record["base"] = read_u64(base_pointer)
    if pitch_pointer is not None:
        record["size"] = read_u64(pitch_pointer) * record["height"]
    live[record["base"]] = record["size"]


def allocation(name):
    record = {"call": name, "event": "allocate", "base_pointer": register("rdi"), "size": register("rsi"),
              "then": allocated}
    if name in PITCHED:
        record.update(pitch_pointer=register("rsi"), height=register("rcx"), size=None)
    capture["allocations"].append(record)
    at_return(record)


def freed(name):
    base = register("rdi")
    capture["allocations"].append({"call": name, "event": "free", "base": base, "known": base in live})
    live.pop(base, None)


ENTRIES = {**{n: module_image for n in MODULE_IMAGES}, **{n: module_file for n in MODULE_FILES},
           **{n: link_data for n in LINK_DATA}, **{n: lookup for n in LOOKUPS},
           **{n: function_handle for n in HANDLE_OF},
           **{n: allocation for n in ALLOCATIONS + PITCHED}, **{n: freed for n in FREES}}


def first_export(*names):
    return next((name for name in names if name in addresses), None)


def driver_range(value):
    """(base, size) of the device allocation the driver says `value` points into, or None."""
    name = first_export("cuMemGetAddressRange_v2", "cuMemGetAddressRange")
    if not name:
        return None
    page = scratch_page()
    if driver_call(name, "int (*)(unsigned long *, unsigned long *, unsigned long)", page, page + 8, value) != 0:
        return None
    return read_u64(page), read_u64(page + 8)


def read_device(base, size):
    """Copy [base, base + size) of device memory to mem/<sha256>.bin; returns its record, or
    {"error": why}. The driver copies into a buffer in the program, CHUNK bytes at a time."""
    name = first_export("cuMemcpyDtoH_v2", "cuMemcpyDtoH")
    if not name:
        return {"error": "the driver exports no cuMemcpyDtoH"}
    if size > MAX_ALLOCATION_BYTES:
        return {"error": f"{size} bytes is over WARPLENS_TRACE_MAX_ALLOCATION_BYTES ({MAX_ALLOCATION_BYTES}); not read"}
    if read_total[0] + size > MAX_TOTAL_BYTES:
        return {"error": f"WARPLENS_TRACE_MAX_TOTAL_BYTES ({MAX_TOTAL_BYTES}) would be passed "
                         f"({read_total[0]} read so far); not read"}
    try:
        if not copy_buffer[0]:
            gdb.execute(r"sharedlibrary libc\.so", to_string=True)
            copy_buffer[0] = int(gdb.parse_and_eval(
                "(unsigned long)((void *(*)(void *, unsigned long, int, int, int, long))mmap)"
                f"(0, {CHUNK}, 3, 0x22, -1, 0)"))
            if copy_buffer[0] in (0, 2 ** 64 - 1):
                copy_buffer[0] = 0
                return {"error": "mmap of the copy buffer in the traced program failed"}
        os.makedirs(os.path.join(OUT, "mem"), exist_ok=True)
        partial = os.path.join(OUT, "mem", "partial")
        digest = hashlib.sha256()
        with open(partial, "wb") as f:
            for at in range(0, size, CHUNK):
                count = min(CHUNK, size - at)
                status = driver_call(name, "int (*)(void *, unsigned long, unsigned long)",
                                     copy_buffer[0], base + at, count)
                if status != 0:
                    return {"error": f"{name} returned {status} for {count} bytes at {base + at:#x}"}
                data = read(copy_buffer[0], count)
                digest.update(data)
                f.write(data)
        read_total[0] += size
        saved = f"mem/{digest.hexdigest()}.bin"
        os.replace(partial, os.path.join(OUT, saved))
        return {"file": saved, "size": size, "sha256": digest.hexdigest()}
    except FAILURES as error:
        return {"error": f"reading {size} bytes at {base:#x}: {error}"}
    finally:
        if os.path.exists(os.path.join(OUT, "mem", "partial")):
            os.unlink(os.path.join(OUT, "mem", "partial"))


def launch_memory_before(launch):
    """Which device memory the launch's parameters point into, and its contents now."""
    if MEMORY_SKIP and re.search(MEMORY_SKIP, launch.get("kernel", "")):
        launch["memory"] = {"skipped": f"the kernel's name matches WARPLENS_TRACE_MEMORY_SKIP ({MEMORY_SKIP})"}
        return
    memory = launch["memory"] = {"pointers": [], "allocations": [], "notes": []}
    if not launch.get("parameters"):
        memory["notes"].append("no parameter of the launch was read; no memory was looked for")
        return
    found = {}          # base -> allocation record
    asked = {}          # value -> (base, size) or None, from the driver
    wide = any(p["size"] >= 64 for p in launch["parameters"])
    gdb.execute("set scheduler-locking " + LOCK_THREADS)
    try:
        for p in launch["parameters"]:
            data = bytes.fromhex(p["hex"])
            for offset in range(0, len(data) - 7):
                value = int.from_bytes(data[offset:offset + 8], "little")
                if not 0x10000 <= value < 1 << 48:      # not an address a process can have
                    continue
                extent = next(((base, size, "followed") for base, size in live.items()
                               if base <= value < base + size), None)
                if extent is None:
                    if value not in asked and len(asked) >= MAX_RANGE_QUERIES:
                        memory["driver_questions_exhausted"] = True
                        continue
                    if value not in asked:
                        asked[value] = driver_range(value)
                    if asked[value] is None:
                        continue
                    extent = (*asked[value], "driver")
                base, size, source = extent
                found.setdefault(base, {"base": base, "size": size, "known_from": source, "referenced": True})
                memory["pointers"].append({"parameter": p["index"], "offset": offset, "value": value,
                                           "allocation": base, "offset_in_allocation": value - base})
        if wide:    # a tensor map may hold its address in a form the scan above does not see
            for base, size in live.items():
                found.setdefault(base, {"base": base, "size": size, "known_from": "followed", "referenced": False})
        memory["live_allocations"] = len(live)
        memory["driver_questions"] = len(asked)
        memory["all_live_allocations_read"] = wide
        for record in sorted(found.values(), key=lambda r: (not r["referenced"], r["base"])):
            if record["known_from"] == "followed" and (said := driver_range(record["base"])):
                record["driver_base"], record["driver_size"] = said     # the driver's own account of it
            record["before"] = read_device(record["base"], record["size"])
            if "error" in record["before"] and not record["referenced"]:
                memory["all_live_allocations_read"] = False
            memory["allocations"].append(record)
    finally:
        gdb.execute("set scheduler-locking off")
    at_return({"then": launch_memory_after, "launch": launch})


def launch_memory_after(record):
    """The launch call has returned: wait for the kernel, then read the same memory again."""
    memory = record["launch"]["memory"]
    memory["launch_result"] = result_of_call()
    gdb.execute("set scheduler-locking " + LOCK_THREADS)
    try:
        if "cuCtxSynchronize" not in addresses:
            memory["notes"].append("the driver exports no cuCtxSynchronize; the kernel may not have finished")
        else:
            memory["synchronize_result"] = driver_call("cuCtxSynchronize", "int (*)(void)")
        for allocation in memory["allocations"]:
            allocation["after"] = read_device(allocation["base"], allocation["size"])
    finally:
        gdb.execute("set scheduler-locking off")


def write_capture(final):
    """capture.json as things stand. Written after every launch, so that a tracer that is
    killed later has left it; a call that has not returned yet has result null."""
    internal = ("then", "handle_pointer", "base_pointer", "pitch_pointer")
    shown = dict(capture, device_bytes_read=read_total[0], unseen_during_calls=list(unseen),
                 other_threads_during_calls="held" if LOCK_THREADS == "on" else "running")
    for group in ("modules", "lookups", "handles", "allocations"):
        shown[group] = [{**{k: v for k, v in record.items() if k not in internal}, "result": None}
                        if "then" in record else record for record in capture[group]]
    if final:
        for _, record in pending.values():
            if "launch" in record:
                record["launch"]["memory"]["notes"].append("the launch call did not return; nothing was read after it")
    with open(os.path.join(OUT, "capture.json"), "w") as f:
        json.dump(shown, f, indent=1)


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


def write_results(launches, final=True):
    os.makedirs(OUT, exist_ok=True)
    if CAPTURE:
        write_capture(final)
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
            key = (gdb.selected_thread().num, register("rsp"))
            name = next((n for n in STOPS if addresses.get(n) == pc), None)
            if name in LAUNCHES:
                try:
                    launches.append(describe_launch(name, len(launches)))
                except (gdb.error, gdb.MemoryError) as error:
                    launches.append({"index": len(launches), "call": name, "note": f"could not be read: {error}"})
                if "memory" in CAPTURE:
                    guarded(launch_memory_before, launches[-1])
                if CAPTURE:
                    write_results(launches, final=False)     # on disk now, whatever happens later
            elif name:
                guarded(ENTRIES[name], name)
            elif pc in return_points and key in pending:
                record = returned(key)
                guarded(record.pop("then"), record)
                if "launch" in record:
                    write_results(launches, final=False)
            go_on()
        else:
            notes.append(f"stopped tracing after {MAX_STOPS} stops")
    except gdb.error as error:
        notes.append(f"gdb: {error}")
    finally:
        write_results(launches)


main()
