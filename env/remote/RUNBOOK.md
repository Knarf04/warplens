# Runbook: the B200 rental

One rented machine, one command, one archive back. This is the checklist for
the person doing it. Everything here is plan steps E19 to E22 (`PLAN.md`,
section 11).

What the statements below rest on is marked: **measured** means on the
development machine (RTX 3080 Ti Laptop, driver 580.173.02, Ubuntu 24.04, on
2026-11-23), **documented** means NVIDIA says so at the link given, **guess**
means neither. Nothing here has run on a B200.

| File | What it does |
| --- | --- |
| `pins.env` | Every version, exact. The only place a version is written. |
| `setup.sh` | Installs the env from the pins and clones the book at its pinned commit. No root, no conda, no system toolkit needed. |
| `preflight.sh` | Is this machine a real B200 and does everything on it work? PASS / WARN / FAIL per check. |
| `run.sh` | Preflight, setup, build and run both probe suites, pack one archive. Resumable. |
| `predict_toolkit.sh` | Makes the predictions again with the installed nvcc, without a GPU. `run.sh` runs it by itself when setup fell back to another toolkit (section 2). |
| `gpu_probe.py`, `check_env.py`, `kernels/` | What preflight runs on the GPU. |

The archive also preserves portable PTX/input/output fixtures and the frozen Python
implementation for future offline ports. Follow [the capture verification and replay
checklist](../../docs/hardware-capture.md) before releasing the instance; a tarball
existing alone does not establish that all cases finished.

## 1. Before renting

Do these at home; none needs the B200.

- [ ] The commit to test contains `validation/blackwell/`, `validation/blackwell_timing/`
      and `env/remote/`. `env/remote/run.sh --dry-run` must show no stage as `MISSING`.
- [ ] The predicted outputs are committed (plan step D17).
- [ ] `python -m pytest -q tests/test_remote_env.py` passes.
- [ ] A dry run of preflight on the development GPU fails for the right reason:
      `WARPLENS_ENV_RUN="conda run -n warplens" env/remote/preflight.sh` must say
      `FAIL gpu-identity ... not a B200`.
- [ ] The rows without a probe dry-run on the development GPU:
      `conda run -n warplens python validation/blackwell/run_rows.py --out /tmp/rows` must
      print `ok` for `formats/ue4m3_ptxas` and `e2e/driver_call_trace.cute_dsl`, and
      `skipped (needs sm_100 ...)` for `e2e/cute_dsl_tma` (section 3, "The rows without a probe").
      The capture of that row dry-runs there with a stand-in kernel:
      `conda run -n warplens python validation/blackwell/e2e/cute_dsl_tma.py --out /tmp/rt11 --script examples/06_cute_dsl/saxpy.py`
      must end with `capture: complete`, and `python -m pytest -q tests/test_launch_capture.py`
      must pass with nothing skipped but the PyTorch test.
      `e2e/driver_call_trace.tirx` is `failed` there for as long as that machine's driver
      is older than the toolkit: it refuses NVRTC's PTX, traced or not.
- [ ] The toolkit policy of section 2 is read, and so is the budget (section 5): 2 hours
      for this rental, 4 more in reserve for a second rental.
- [ ] The image is checked against "Before paying" in section 2: counters, ptrace, driver,
      network, disk.
- [ ] Write down the commit: `git rev-parse HEAD`. The archive's manifest must show the same.
- [ ] Make the bundle to upload (section 3). No credentials go on the rented machine.

## 2. Choosing an instance

**It must be a B200, compute capability 10.0.** NVIDIA's table
(<https://developer.nvidia.com/cuda/gpus>, read 2026-11-23, documented):

| Card | Compute capability | Usable |
| --- | --- | --- |
| NVIDIA B200, NVIDIA GB200 | 10.0 (`sm_100a`) | B200 yes; GB200 see below |
| NVIDIA B300 | 10.3 (`sm_103a`) | no: not the target the probes are built for |
| GeForce RTX 5090 / 5080 / 50 series | 12.0 | no |
| NVIDIA RTX PRO 6000 Blackwell (Workstation **and Server Edition**) | 12.0 | no |
| NVIDIA GB10 (DGX Spark) | 12.1 | no |
| H100, H200 | 9.0 | no |

### Before paying: what to confirm about the image

A listing that says B200 can still be an image on which the run stops at preflight or comes
back without what it was rented for. Each line is something a script here needs; where it
says so is in brackets. Ask the provider, or read the image's description, before the clock
starts.

- [ ] **Root on a VM or bare metal, or a container started with `CAP_SYS_ADMIN`.** `ncu` reads
      performance counters only as root or with `CAP_SYS_ADMIN` (`CAP_PERFMON` also works
      from driver R565), or when the *host's* driver has
      `NVreg_RestrictProfilingToAdminUsers=0` (documented:
      <https://developer.nvidia.com/ERR_NVGPUCTRPERM>). Root *inside* a container is not
      enough: only the provider can start it with `--cap-add=SYS_ADMIN` or open the host
      driver. Without a counter read the full preflight FAILs and `run.sh` stops before
      anything is built or run on the GPU (`preflight.sh`, check `ncu`); the timing model
      is held against counters, so this is asked first. The alternative is decided at home,
      not on the clock: `run.sh --no-ncu` (section 7).
- [ ] **`ptrace` is permitted in the container.** Rows RT-10 and RT-11 start a program under
      gdb (`validation/blackwell/e2e/driver_call_trace.py`); where the container's seccomp
      profile forbids `ptrace` they come back `skipped (... ptrace is not permitted)`, and
      the tensor maps of RT-11 are a reason for a second rental (section 5). Docker's
      default profile allows it on current kernels (not tried in a provider's container:
      section 9); a VM or bare metal has no such profile. The run does not stop for this,
      so nothing warns you before the rows are lost: ask.
- [ ] **The driver's "CUDA Version" is at least the pinned toolkit's** (`CUDA_VERSION` in
      `pins.env`: 13.5). If the image has an older 13.x driver the run still works, with the
      fallback toolkit ("Driver" below): then make that toolkit's predictions at home
      *before* renting ("At home: the predictions of a fallback toolkit"), so that what
      depends on the compiler is known in advance. Below 12.8 (`CUDA_VERSION_MIN`) preflight
      fails. A rented driver cannot be changed from inside a container.
- [ ] **Outbound HTTPS**, to: github.com and the hosts its downloads redirect to (the
      micromamba binary, the clone of the book, the clone of CUTLASS that
      `examples/03_cutlass_mma/build.sh` makes for the end-to-end step, and the clock check);
      conda.anaconda.org (conda-forge: Python, nvcc, the CUDA headers, NVRTC, Nsight Compute;
      git and gdb too when the image has none); pypi.org and files.pythonhosted.org (Triton,
      the CuTe DSL, TIRx, cuda-bindings, PyTorch). And the image's own package mirror if
      you are root and it lacks `curl`, `git` or `gdb` (`apt-get`). No fetch here can be
      replaced by an upload except the repository itself (the bundle of section 3) and,
      with `WARPLENS_BOOK`, the book.
- [ ] **18 GB free on one filesystem you can write to** (`WARPLENS_REMOTE_HOME`; `setup.sh`
      refuses to start with less, 12 GB with `--no-torch`), and not counting `/tmp`:
      everything, temporary files included, goes under that directory. The archive is small
      next to it (section 6).
- [ ] **x86_64, one dedicated GPU.** `setup.sh` refuses any other CPU (a GB200 node is Arm);
      a shared card makes the timing numbers worthless and preflight cannot always see it.

What to look for in a listing:

- The word "Blackwell" is not enough. The RTX PRO 6000 Blackwell Server Edition
  is sold as a datacenter Blackwell card and is 12.0; it has no `tcgen05` and no
  tensor memory. The listing must say **B200**.
- Listings seen on 2026-11-23 name it "B200", "1x B200 (SXM6)", "B200 SXM6" or
  "HGX B200" (third-party price trackers, not verified with the providers:
  <https://www.beam.cloud/blog/nvidia-b200-pricing>,
  <https://www.spheron.network/blog/nvidia-b200-cloud-pricing-2026/>).
- **Avoid GB200.** It has the same GPU, but a GB200 node has Arm CPUs.
  `setup.sh` refuses anything but x86_64: the pinned micromamba checksum and the
  PyPI wheels were only tried there.
- **One GPU is enough.** Some providers rent B200 only as a whole 8-GPU node; that
  multiplies the price by eight for nothing.
- **Prefer a VM or bare metal where you are root over a container** ("pod").
  `ncu` reads performance counters only as root (or with `CAP_SYS_ADMIN`), or
  when the *host's* driver was configured to allow everyone
  (documented: <https://developer.nvidia.com/ERR_NVGPUCTRPERM>). In a VM you are
  root over the driver. In a container you are root only inside it: unless the
  provider started it with `--cap-add=SYS_ADMIN` or opened the host driver,
  nothing you do inside helps. **A full run requires a counter read**: without
  one the full preflight FAILs (check `ncu`) and `run.sh` stops before anything
  is built or run on the GPU. The run goes on without counters only when told
  to, with `run.sh --no-ncu`; the `ncu` comparison in the timing suite (plan
  step C14) is then lost and the archive says so (section 7).
- **Driver.** Ask for, or pick the image with, the newest driver. What matters is
  the "CUDA Version" in the header of `nvidia-smi`:
  - 13.5 or newer: everything runs as pinned, and results are held against the
    committed predictions without reservation.
  - 13.0 to 13.4: `setup.sh` installs that toolkit release in place of 13.5
    (measured on a 13.0 driver: PTX from nvcc 13.5 is refused with
    `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, machine code from it runs, and with
    toolkit 13.0 installed both run). **The probes are then not built by the
    compiler the committed predictions were made with**; the run says so
    everywhere and makes the predictions again with the installed compiler
    before it touches the GPU. The policy is decided: "Driver older than the
    toolkit" below. The same command either way.
  - 12.8 or 12.9: the CUDA C++ probes can run with toolkit 12.x, but TIRx and the
    CuTe DSL will not (measured with toolkit 12.8: their tests fail; they need
    `libcudart.so.13`), and there is no PyTorch for the Triton step (the pinned
    PyTorch carries the CUDA 13.2 runtime). Avoid if there is a choice.
  - below 12.8: unusable; preflight fails.
- **Disk:** 18 GB free where `WARPLENS_REMOTE_HOME` points (measured: 9.7 GB
  after `setup.sh`, of which the PyTorch env is 4.3 GB; downloads and unpacking
  need the rest for a while). `setup.sh` puts its temporary files there too,
  not in `/tmp`. **Network:** outbound HTTPS to github.com, conda.anaconda.org
  and pypi.org.
- A shared card (another tenant on the same GPU) makes the timing numbers
  worthless. Preflight warns when it sees other processes or load, but from
  inside a container it cannot see other tenants. Rent a dedicated GPU.

### Driver older than the toolkit

**Decided on 2026-11-24, so that nothing is left to decide on the day: the pinned
toolkit 13.5 when the driver takes its PTX; otherwise the toolkit release that matches
the driver, with the predictions made again on the spot by that compiler, before
anything runs on the GPU.** `env/remote/run.sh` with no option does exactly this.

The procedure, by the "CUDA Version" in the header of `nvidia-smi`
(`env/remote/preflight.sh --quick` prints it as the `driver` check):

| Driver's CUDA Version | What to type | What happens |
| --- | --- | --- |
| 13.5 or newer | `env/remote/run.sh` | `setup.sh` installs nvcc 13.5.37, the predictions' compiler. `predictions_comparable=comparable`; stage `predict-toolkit` is skipped. Results are held against the committed predictions. |
| 13.0 to 13.4 | `env/remote/run.sh` (the same) | The fallback, steps 1 to 6 below. |
| 12.8 or 12.9 | `WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e env/remote/run.sh` | Avoid such a machine. If there is no other: toolkit 12.x, probes and timing only (no TIRx, no CuTe DSL, no PyTorch). `predict-toolkit` is attempted and may fail, because the simulator's launch paths are built for CUDA 13; not tried. |
| below 12.8 | nothing | Release the instance: preflight fails. |

The fallback on a 13.0 to 13.4 driver, each step done by `run.sh` without being asked:

1. `setup` installs the conda-forge toolkit of the driver's release (on the development
   machine's 13.0 driver: nvcc 13.0.88) and says so between rows of `!`;
   `resolved/versions.txt` has `predictions_comparable=not-comparable`.
2. `selftest` runs the whole predict path with that nvcc against the committed
   predictions (`tests/test_predict_path.py`); its failures name the predictions that
   do not hold for this compiler.
3. `build-blackwell` and `build-timing` build with that nvcc.
4. `predict-toolkit` (`env/remote/predict_toolkit.sh`) runs every probe and every timing
   benchmark in the simulator, from the programs just built, into
   `out/predicted-toolkit/blackwell/` and `out/predicted-toolkit/timing/` of the archive,
   and lists in `out/predicted-toolkit/DIFFERENCES.txt` the predictions that differ from the
   committed ones. It runs **before** the GPU stages, so a prediction is still on record
   before the measurement it is held against, which is the point of plan step D17. No
   committed file is touched.
5. `run-blackwell` and `run-timing` run on the GPU. The driver compiles the PTX the
   simulator ran in step 4: the same compilation on both sides.
6. At home, hold the results against the predictions of step 4, not the committed ones:

   ```bash
   tar xzf warplens-b200-*.tar.gz && cd warplens-b200-*/
   cat out/predicted-toolkit/toolkit.txt out/predicted-toolkit/DIFFERENCES.txt
   python <checkout>/validation/blackwell/compare.py out/blackwell --predicted out/predicted-toolkit/blackwell
   python <checkout>/validation/blackwell_timing/fit.py out/timing --predicted out/predicted-toolkit/timing
   ```

   `compare.py` prints no toolkit banner then: it reads `provenance.json` of the
   directory it is given, which names the rental's nvcc. `PREDICTIONS-NOT-COMPARABLE.txt`
   at the top of the archive repeats these commands. The `fit/` that the timing suite
   writes on the rented machine used the committed predictions and is flagged; ignore it.

If `predict-toolkit` fails or times out (30 minutes), the run goes on and the archive
says so (`stage_predict_toolkit` in `manifest.txt`). Make the predictions at home with
the commands under "At home" below, and state in the write-up that they were made after
the measurement.

Everything is flagged while it happens, so the fallback cannot pass unnoticed:
`setup.sh` (a block between rows of `!` at the end), `preflight.sh` (check `predictions`,
WARN), `run.sh` (after setup and as its last output; `predictions_comparable` in
`manifest.txt`; `PREDICTIONS-NOT-COMPARABLE.txt`), `validation/blackwell/run_all.py`
(`TOOLKIT-MISMATCH.txt`, `env/toolkit.txt`), `compare.py` without `--predicted` (before and
after its table), and the timing suite's `TOOLKIT.txt` and `status.txt`.
`run.sh --strict-toolkit` stops after setup instead of going on.

**Why the fallback and not `--toolkit pinned`.** The two ways to run on such a driver:

| | `--toolkit auto` (the default, and the policy) | `--toolkit pinned` |
| --- | --- | --- |
| nvcc installed | the release matching the driver (on the development machine: 13.0.88) | 13.5.37, as pinned |
| probes run as | PTX, compiled by the driver | machine code (`bin-native/`), compiled by the toolkit's ptxas; `run_all.py` chooses it by itself (`choose_binaries`: it runs `mbarrier/script basic` from `bin/` and, if that fails and the one in `bin-native/` works, uses `bin-native/`) |
| same compiler as the committed predictions | **no**: `BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE`; made again by `predict-toolkit` | yes (`predictions_comparable=comparable-native`) |
| end-to-end programs | all run | the CUTLASS tutorials (programs that carry PTX) and TIRx (NVRTC 13.5 writes PTX) are refused by the driver; preflight fails `e2e-ptx`, so the run needs `WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e` and rows RT-06 to RT-09 are lost |
| what the GPU executes | the PTX the simulator ran, through the driver's compiler | the same source through ptxas 13.5; the driver's compiler is not exercised |
| timing benchmarks | built by the other nvcc; predictions made again for it | as predicted |
| ncu | the release for that toolkit (measured: 2025.3.1.4 with 13.0) | 2026.4.0.10, as pinned |

The fallback loses nothing but the committed predictions, and those can be made again
without a GPU; `pinned` loses four end-to-end rows that cannot be recovered afterwards.

**How the predictions depend on the compiler** (what the choice rests on):

- A probe prediction is what the probe printed under `warplens run`, which executes the
  PTX that nvcc put into `validation/blackwell/bin/<probe>`. The instructions under test
  are inline PTX and pass through nvcc unchanged; the compiler decides the code around
  them (loading arguments, loops, branches, registers). The output is values, not code,
  so it changes with the compiler only if a probe prints something that depends on code
  shape, or the simulator mishandles a construct one compiler emits and the other does
  not. Measured in the rehearsal of 2026-11-23 with nvcc 13.0.88: all 98 probe
  predictions byte-identical to those of 13.5.37, after one such simulator bug was fixed
  (`validation/blackwell/README.md`, finding 16: 16 predictions differed before the fix).
  That is a measurement for 13.0.88, not a guarantee for 13.1 to 13.4.
- A timing prediction is instruction counts and modelled times of the compiled code, so
  it is the compiler's: with 13.0.88, 13 of the 14 benchmarks had other instruction counts.
- The end-to-end predictions (`predicted/e2e/`) are not made again by `predict-toolkit`:
  `compare.py` does not read them, and the Triton and CuTe DSL kernels are not compiled
  by nvcc.
- Which nvcc made a set of predictions is in its `provenance.json` (probes) and
  `meta.json` (timing); `run_all.py`, `compare.py` and the timing suite compare it with
  the nvcc that built the programs.

**At home: the predictions of a fallback toolkit, before or after the rental.** They need
no GPU. Before renting, when the listing names the driver, this shows in advance which
predictions depend on the compiler; after, it stands in for a failed `predict-toolkit`.

```bash
# X.Y: the driver's CUDA Version (after a rental: toolkit_installed in resolved/versions.txt)
export WARPLENS_REMOTE_HOME=$HOME/warplens-X.Y        # 12 GB
env/remote/setup.sh --toolkit X.Y --no-torch --no-ncu
env/remote/predict_toolkit.sh --build --out $HOME/predicted-X.Y
cat $HOME/predicted-X.Y/toolkit.txt $HOME/predicted-X.Y/DIFFERENCES.txt
python validation/blackwell/compare.py <the run's out/blackwell> --predicted $HOME/predicted-X.Y/blackwell
python validation/blackwell_timing/fit.py <the run's out/timing> --predicted $HOME/predicted-X.Y/timing
```

`conda-forge` may serve another patch release of X.Y than the rental got;
`resolved/conda-explicit.txt` in the archive has the exact packages. Not run on
2026-11-24: no toolkit between 13.0 and 13.4 is installed on the development machine
(the `warplens` conda env has nvcc 13.5.37; `/usr/local/cuda` is 12.4.131, which does
not know `sm_100a`; the 13.0 env of the rehearsal was removed with its directory).
`predict_toolkit.sh --build` was run here with nvcc 13.5.37 only, where it must and does
reproduce the committed predictions (section 9).

**The one deviation worth its time.** If `out/predicted-toolkit/DIFFERENCES.txt` lists probe
predictions under `[blackwell]` (none are expected) and the clock allows about 15 more
minutes, run the probes a second time with the predictions' compiler into a second env,
and bring back both archives:

```bash
WARPLENS_REMOTE_HOME=$HOME/warplens-pinned WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e \
  env/remote/run.sh --toolkit pinned --skip selftest
```

Only the default was run end to end here (on the 3080 Ti, where everything Blackwell
fails by design); `--toolkit pinned` was measured for setup, preflight and the PTX
refusal, not for a whole run.

## 3. From a blank machine to results

On your own machine, pack the repository (history included, so the commit is recorded):

```bash
git bundle create warplens.bundle HEAD
scp warplens.bundle <user>@<host>:
```

On the rented machine. Use `tmux` (or `nohup`) so that a dropped connection
does not matter; `run.sh` also ignores the hang-up itself.

```bash
nvidia-smi                                   # first look: is it a B200, which CUDA Version
git clone warplens.bundle warplens && cd warplens     # no git? see below
export WARPLENS_REMOTE_HOME=$HOME/warplens-remote     # 18 GB; point it at the big disk
env/remote/preflight.sh --quick              # 2 seconds. FAIL gpu-identity: release the machine now
env/remote/run.sh --dry-run                  # the plan; no stage may say MISSING
tmux new -s run
env/remote/run.sh 2>&1 | tee ~/run.log       # everything; prints the archive path and checksum at the end
```

The same command whatever the driver is: the toolkit policy of section 2 is built in.
Nothing else is fetched by hand: `setup.sh` installs PyTorch and clones the book
("PyTorch and the book" below). A full run needs `ncu` to read a counter and stops at
preflight if it cannot (section 7, `ncu` FAIL): become root before the first command.

- No `git` on the machine: `setup.sh` installs it with `apt-get` when it is root
  (not exercised) or puts conda-forge's git into the env (measured). To get the
  checkout there without git, copy the tree instead:
  `rsync -a --exclude __pycache__ ./ <user>@<host>:warplens/` (include `.git`;
  the commit is read from it without a git program).
- If you are not root and `sudo` works, become root first (`sudo -i`) and do
  everything as root, so that `ncu` may read counters; without them the run
  stops at preflight. Decide this before the first command: the env lives under
  `$HOME` of whoever runs `setup.sh`.
- Again after any failure or interruption: the same `env/remote/run.sh`.
  Finished stages are skipped only for the same commit, pins and run configuration.
  Changing `--toolkit` or suite arguments reruns the stages. `--redo STAGE` repeats
  one and invalidates dependent results: rebuilding a suite reruns that suite;
  rerunning setup repeats preflight, builds and runs. `--skip STAGE` leaves one
  out; `--keep-going` continues past a failed preflight.

Back on your own machine (section 6):

```bash
scp '<user>@<host>:warplens-remote/results/warplens-b200-*' .
sha256sum -c warplens-b200-*.tar.gz.sha256
```

### PyTorch and the book

**PyTorch: `setup.sh` installs it; there is nothing to prepare.** What it does
(`setup.sh` section 5b, `install_torch_env` in `lib.sh`, the pins at the end of `pins.env`):

- After the main env it makes a venv, `$WARPLENS_REMOTE_HOME/torch-env`, with
  `python -m venv --system-site-packages` on the main env's Python, and installs
  `torch==2.15.0` and `triton==3.9.0` into it from PyPI, every package torch brings held
  to its `TORCH_PIP_` pin. PyPI's Linux wheel of 2.15.0 is the CUDA 13.2 build
  (`2.15.0+cu132`). The venv sees the main env's packages (Triton, NumPy, the CuTe DSL).
- It is not installed into the main env on purpose: torch's CUDA 13.2 wheels there make
  `cuda-bindings` load NVRTC 13.2 in place of the toolkit's, which changes the PTX of
  every TIRx kernel (measured; `pins.env`).
- The pair that was tried: torch 2.15.0+cu132 with Triton 3.9.0 (the Triton this torch
  release requires, and the one the simulator's tests ran with), on the development
  machine's driver 580.173.02 (CUDA 13.0), where `validation/blackwell/e2e/triton_gpu.py`
  ran whole on the RTX 3080 Ti. On a B200 it is untried: whether this wheel was built
  for `sm_100` is what `torch_arch_list` in `resolved/versions.txt` will say.
- It is skipped, with a warning, in two cases: `--no-torch` was given, or the driver is
  of the 12.x series (it cannot load the CUDA 13 runtime the wheel carries, and no torch
  release pairs Triton 3.9.0 with CUDA 12). A failed install (network, disk) is a warning
  too, and every later `setup.sh` tries again.
- What uses it: the Triton-on-GPU step (`e2e/triton`, row RT-07) and the CuTe DSL
  tutorial of row RT-11. `run.sh` hands the suites its interpreter as
  `WARPLENS_TORCH_PYTHON`. `resolved/versions.txt` records `torch`, `torch_build`,
  `torch_cuda_version`, `torch_cuda_available` and `torch_device_cap`.
- It takes 4.3 GB and 144 s on a fast link (measured).

**The book: `setup.sh` fetches it; there is nothing to prepare.** The book's kernels (row
RT-08) are run from the book's own repository, at the commit the predictions were made
against. It has no license, so it is not in this repository and is only ever cloned.
What `setup.sh` does (its section 5d, `ensure_book` in `lib.sh`, `BOOK_COMMIT` in `pins.env`):

- It clones <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys> into
  `$WARPLENS_REMOTE_HOME/book`, that one commit without history (11 MB; measured), and
  checks that what it got is at exactly the pinned commit and has the GEMM chapter. The
  pin is the commit `tests/test_book.py` names (`COMMIT`) and CI checks out;
  `tests/test_remote_env.py` holds the three together.
- Every run of `setup.sh` looks again: a clone at the pinned commit is left alone, one at
  another commit is replaced, a missing one is fetched. The fetch goes into a directory
  beside it and takes its place only when it is right, so a failed fetch leaves what was
  there.
- A failed fetch (no network, the commit gone) is a warning, as for PyTorch, and the next
  `setup.sh` tries again; `book_commit=` in `resolved/versions.txt` says what is there.
- It is not fetched with `setup.sh --no-book` (through `run.sh`: `run.sh --no-book`), or
  when `WARPLENS_BOOK` is set: that clone is then the one used.
- `run.sh` and `preflight.sh` take `$WARPLENS_REMOTE_HOME/book` for `WARPLENS_BOOK` when the
  variable is not set and the clone is there. `WARPLENS_BOOK` set by you wins, and so does
  `--book` (through `WARPLENS_BLACKWELL_RUN_ARGS` for `run.sh`); `WARPLENS_BOOK=` set to
  nothing means no book. `book=` and `book_commit=` in `manifest.txt` say which was used.
- The clone lies beside the run directory, not in it, and `run.sh` archives the run
  directory only: no file of the book is in the archive. What is there, as before, is
  what the captures keep of a run: the PTX compiled from its kernels, the commit and a
  checksum of each chapter read.

The fallback, when the rented machine cannot reach github.com but you can upload: clone it
by hand, at the same commit, and name it. These are the commands CI runs.

```bash
git init -q ~/book
git -C ~/book fetch -q --depth 1 https://github.com/mlc-ai/modern-gpu-programming-for-mlsys 3a24144defa6da03d4ab8a1dfcf513b18007d15f
git -C ~/book checkout -q FETCH_HEAD        # elsewhere, then: rsync -a ~/book <user>@<host>:
export WARPLENS_BOOK=$HOME/book             # before env/remote/run.sh
```

A full run checks both before the suites start: the preflight check `native-e2e` FAILs,
and the run stops, when the PyTorch env does not see the GPU or there is no clone of the
book; the check `book` says which commit the clone is at (FAIL for `setup.sh`'s own clone
at another commit than the pinned one, WARN for a clone you named). `run.sh` asks for that check by itself unless the Blackwell suite is
skipped or restricted with `--no-e2e` / `--only`; a standalone `preflight.sh` does it
with `--e2e`. If PyTorch cannot be had (a 12.x driver), run the probes without the
end-to-end programs, which also leaves out the launches of rows RT-10 and RT-11. That is a
run without the RT-11 capture by request, and not a failure; the capture is then still
owed (section 5):

```bash
WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e env/remote/run.sh
```

### The rows without a probe: NUM-05, RT-10, RT-11

Three rows of `docs/validation-matrix.md` are not kernels with a predicted output. They
are scripted all the same: `validation/blackwell/run_all.py` ends a GPU run with
`validation/blackwell/run_rows.py`, inside the stage `run-blackwell`, so there is nothing
to type. Outputs go to `out/blackwell/rows/` of the archive, one line per step to
`out/blackwell/rows/status.tsv`. NUM-05 and RT-10 are informational: a step that fails or
is skipped is a line there and never fails the stage.

**RT-11 is required on a B200.** Its capture is the one thing of the rental that nothing
else produces and that a later decoder cannot be written without, so on a GPU of compute
capability 10.x anything but `ok` for `e2e/cute_dsl_tma` (skipped for want of gdb or
PyTorch, failed, incomplete, timed out) makes `run_rows.py`, `run_all.py` and so the stage
`run-blackwell` fail. It is said in a banner that `run_rows.py`, `run_all.py`, `run.sh`
(after its line of stage statuses) and `compare.py` each print, and in
`out/blackwell/RT-11-NOT-CAPTURED.txt`, which is there for exactly as long as the capture is
missing. On any other GPU the step skips itself and nothing fails. `--no-e2e` and `--only`
(through `WARPLENS_BLACKWELL_RUN_ARGS`) leave the rows out and are not failures;
`--rt11 optional` there runs the row and accepts whatever it ends as.

| Row | Step in `rows/status.tsv` | What runs | What to look for afterwards |
| --- | --- | --- | --- |
| NUM-05 (UE4M3) | `formats/ue4m3_ptxas` | `formats/ue4m3_ptxas.py`: asks the installed ptxas whether any `cvt` to or from UE4M3 exists, with the formats that do exist as controls. No GPU | `rows/formats/ue4m3_ptxas.stdout` ends with `NUM-05: no UE4M3 conversion in this toolkit; the row stays open`. Status `news (...)` means ptxas accepted one: the row can then be tested with a sweep like NUM-01's |
| RT-10 (driver calls) | `e2e/driver_call_trace.tirx`, `e2e/driver_call_trace.cute_dsl` | `e2e/driver_call_trace.py`: `examples/07_tirx/run.py` and `examples/06_cute_dsl/saxpy.py` under gdb, a breakpoint on every export of `libcuda.so` and `libcudart.so` | `rows/e2e/driver_call_trace.<name>/summary.txt` (each call and how often) and `calls.txt` (in order) against the table of `docs/m6.md`; a call the shims lack is the news |
| RT-11 (a CuTe DSL kernel with TMA: tensor maps, code, memory) | `e2e/cute_dsl_tma` | `e2e/cute_dsl_tma.py`: CUTLASS's `fp16_gemm_0.py`, unmodified, once plain, once under the same tracer, which reads every kernel parameter at the launch, and once more under it at a small size with the capture on: the code handed to the driver, the DSL's PTX, and the device memory the launch's parameters point into, before the kernel and after it. Then `e2e/launch_capture.py` checks that nothing is missing | `rows/e2e/cute_dsl_tma/summary.txt` starts with `RT-11 capture: COMPLETE`; the table below says what is where. **This is what unblocks CuTe DSL kernels with TMA in the simulator, and the only required row here: without it the stage fails** (below, and section 8) |

What they need, and what `setup.sh` does about it: gdb with Python support. `setup.sh`
uses the machine's gdb, else installs the system package as root, else puts
conda-forge's gdb into `$WARPLENS_REMOTE_HOME/gdb-env` (0.3 GB, not pinned: it is a tool,
not part of what is measured); `gdb=` in `resolved/versions.txt` says which. RT-11 also
needs the PyTorch env and the CUTLASS clone that the end-to-end step makes.

If a step says `skipped` or `failed` on the B200, run it by hand before releasing the
instance; each takes under a minute. From the checkout, with `E=$WARPLENS_REMOTE_HOME/envrun`
and `R=$WARPLENS_REMOTE_HOME/run/out/blackwell`:

```bash
cat $R/rows/status.tsv                                  # which step, and why
cat $R/rows/e2e/cute_dsl_tma/summary.txt                # RT-11: what was captured and what is missing
command -v gdb || apt-get install -y gdb                # the usual cause; or WARPLENS_GDB=/path/to/gdb
export WARPLENS_TORCH_PYTHON=$WARPLENS_REMOTE_HOME/torch-env/bin/python   # RT-11's PyTorch; run.sh sets it itself
$E python validation/blackwell/run_rows.py --out $R     # all of them again
# or one at a time (status.tsv keeps the lines of the steps that are not run again):
$E python validation/blackwell/run_rows.py --out $R --only e2e/cute_dsl_tma
$E python validation/blackwell/run_rows.py --out $R --only 'e2e/driver_call_trace.*'
$E python validation/blackwell/run_rows.py --out $R --only formats/ue4m3_ptxas
env/remote/run.sh --collect-only                        # pack again; the new files are in the archive
```

Run RT-11 through `run_rows.py`, as above, and not `cute_dsl_tma.py` directly: it is
`run_rows.py` that takes `RT-11-NOT-CAPTURED.txt` away when the row ends as `ok`, and
`run.sh --collect-only` exits 1 and repeats the banner for as long as that file is there.
The stage `run-blackwell` stays `failed` in `manifest.txt` (it did fail); the capture's own
`summary.txt` saying `RT-11 capture: COMPLETE` is what counts afterwards.

What RT-11 leaves in `rows/e2e/cute_dsl_tma/`, and what to do when its capture run is the
part that fails:

| Path | What |
| --- | --- |
| `summary.txt` | first line `RT-11 capture: COMPLETE` or `INCOMPLETE`; then every problem, every warning, and each pointer with its allocation and its two dumps |
| `capture.sha256` | SHA-256 of every file here; `sha256sum -c capture.sha256` |
| `tutorial.stdout` | the tutorial run plain, at its own size (8192 x 8192 x 8192); ends with `PASS` |
| `trace/` | the same under the tracer: `launches.txt`, `launches.json`, `launchNN.paramMM.bin`. The tensor maps at the tutorial's own size |
| `capture/` | the tutorial once more under the tracer, at `--mnk 256,512,512`, with everything a replay needs: `moduleNN.bin` (the fat binary the DSL hands the driver), `dsl_dump/*.ptx` and `*.cubin` (the DSL's own dump of that compilation; the PTX is what the simulator runs), `capture.json` (modules, kernel lookups, allocations), `launches.json` (parameters, and under `memory` each pointer found in them with its allocation and dumps), `mem/<sha256>.bin` (device memory before the kernel and after it), `capture.log` (every call the tracer made in the program, written before the call), `command.txt`, `expect.json` |

About 5 MB: the three tensors are 1 MiB together and PyTorch keeps them in one 2 MiB
allocation, which is saved whole before and after the kernel (measured for a stand-in
kernel over PyTorch tensors on an RTX 3080 Ti, where the tutorial itself got as far as
that one allocation; not for the tutorial's launch). `python validation/blackwell/e2e/launch_capture.py verify $R/rows/e2e/cute_dsl_tma`
checks it again, there or at home.

- `INCOMPLETE` with `the traced run did not print 'PASS'` and a failed assertion in
  `capture/gdb.stdout`: the tutorial does not pass at the small size. Take another:
  `$E env WARPLENS_RT11_CAPTURE_MNK=512,512,512 python validation/blackwell/run_rows.py --out $R --only e2e/cute_dsl_tma`
  (M a multiple of 128, N of 256. An empty value is the tutorial's own size: three
  allocations of 128 MiB, 0.8 GB of dumps).
- The capture run times out, and the last line of `capture/capture.log` is a call with no
  result under it: that call into the driver did not return. The tracer makes its calls with
  the program's other threads running; try them held, `WARPLENS_TRACE_LOCK_THREADS=1` in
  front of the command (on driver 580.173.02 one call needed the opposite, which is why
  running is the default). What was saved before the hang is kept.
- `WARNING: ... no pointer was found in parameter N (128 bytes)` is not a failure: the
  address in that tensor map is in a form the scan does not know, and every live
  allocation was saved instead.
- `no PTX with .entry ...`: the DSL's own dump did not appear in `capture/dsl_dump/`. The
  machine code in `moduleNN.bin` is still the kernel; say so in the notes and keep the
  directory. A replay needs the PTX, so this is a reason to look at
  `CUTE_DSL_KEEP`/`CUTE_DSL_DUMP_DIR` in `capture/command.txt` before leaving.

Still by hand, and optional (matrix row RT-11, last sentence): the DSL's experimental
`create_tensor_map_tiled` (`cutlass.experimental.cuda.tensor_map`), which `docs/m6.md`
says may ask the driver for its tensor map instead of writing one. No script uses it,
because it needs a kernel written against an experimental API that has not run anywhere
here. If someone writes one, trace it the same way and keep the directory:
`driver_call_trace.py --out $R/rows/e2e/create_tensor_map_tiled -- python <script>`;
a `cuTensorMapEncodeTiled` in its `summary.txt` settles the question.

What was and was not tried of this is in section 9.

## 4. What each stage takes

`env/remote/run.sh --dry-run` prints this table for the stages still to run.

| Stage | Minutes | Basis |
| --- | --- | --- |
| log in, upload, clone | 5 to 10 | guess |
| `preflight-quick` | under 1 | measured: 2 s |
| `setup` | 3 to 12 | measured: 79 s for the main env and 144 s for the PyTorch env on a fast link (9.7 GB on disk in all); the upper end is a guess for a slow link or disk |
| `preflight` | 1 to 2 | measured: 5 s, but there the Blackwell kernels fail at load; on a B200 they run, and `ncu` attaches |
| `selftest` (the repository's tests, in the simulator) | 4 to 20 | measured: 10.5 min on 20 cores (1513 tests). It includes the whole predict path, which is what says whether the committed predictions hold for the installed compiler; worth its time on a first rental with a fallback toolkit, `--skip selftest` otherwise. Limit: 30 min |
| `build-blackwell` | 1 to 6 | measured: 8 s on 20 cores for the 51 programs (13 s with `--native`); the upper end is a guess for a smaller machine |
| `build-timing` | 1 to 4 | measured: 62 s for the 14 benchmarks |
| `predict-toolkit` | 0, or 5 to 15 | 0 when the driver takes the pinned toolkit's PTX (the stage is skipped). With a fallback toolkit, measured with nvcc 13.5.37 on 20 cores: 65 s for the 108 probe predictions, 8 at once, and about 7 min for the 14 timing benchmarks, one at a time; the upper end is a guess for a slower CPU. Limit: 30 min |
| `run-blackwell` | 3 to 20 | guess. Kernels are tiny; the time is process start-up and the probes that wait out a timeout on purpose. The rows without a probe at its end add about 15 s here (measured without the RT-11 tutorial, which needs a B200; that is three runs of a GEMM, two of them under gdb, and a guess of 1 to 3 min; a stand-in over PyTorch tensors took 7 s under the tracer with the capture on) |
| `run-timing` | 3 to 20 | guess. Under `ncu` every kernel is replayed per counter group |
| `collect` | under 1 | measured |
| copy back, verify, second copy | 5 | guess |
| **Total** | **28 to 116** | the two run stages are the unknown; 101 at the upper end when the driver takes the pinned toolkit; 20 fewer with `--skip selftest`, which a fallback run on a tight clock can afford, since `predict-toolkit` then says which predictions depend on the compiler |

Limits: each build stage is stopped after 30 minutes and each run stage after 60
(`WARPLENS_BUILD_MINUTES`, `WARPLENS_RUN_MINUTES`). The timing suite gives each
benchmark 15 minutes of its own, so four hung benchmarks would reach the stage
limit; raise `WARPLENS_RUN_MINUTES` if the first minutes show benchmarks hanging.

## 5. Cost

Assumption: **$8 per GPU-hour**, on demand. Third-party trackers in November 2026
showed a median near $8 and single-GPU offers from about $4 to $8, with
whole-node offers far higher per hour (links in section 2; not verified with
the providers, and prices move).

| | Hours | At $8/h |
| --- | --- | --- |
| Everything works first time | 0.5 to 1.9 | $4 to $16 |
| Budget, with one retry of a stage | 2 | $16 |
| Provider bills whole hours | 1 or 2 | $8 or $16 |
| Only an 8-GPU node is on offer | 2 x 8 GPUs | about $130 |

### A second rental (plan step F24)

**Decided by the owner on 2026-11-24: 6 GPU-hours on a single B200 in all. 2 hours for the
first rental, the other 4 held in reserve for what comes after it.**

| | Hours | At $8/h | From |
| --- | --- | --- | --- |
| **First rental** | **2** | **$16** | the budget line of the table above |
| A second run, everything works | 0.4 to 1.6 | $3 to $13 | section 4 without `selftest`: 24 to 96 minutes (to 81 when the driver takes the pinned toolkit) |
| **Reserve, for later rentals** | **4** | **$32** | 6 - 2: two more runs at the first rental's budget line, or one with time to look at it |
| **Ceiling for everything** | **6** | **$48** | the owner's number |

- The budget is in GPU-hours; the dollars are those hours at the assumed $8. At another
  price the hours stand and the owner is told what they cost before the clock starts.
- The first rental stops at 2 hours whatever the reserve is: what the reserve is for is a
  run after a fix, not a longer first run.
- An 8-GPU node (about $130 for two hours) is outside this budget for any rental: it
  spends eight GPU-hours an hour and needs the owner's explicit yes each time.
- What stops a rental at its ceiling is section 8: Ctrl-C packs what exists.

What triggers the second rental. Any one of these, and only after the fix is committed,
the predictions are regenerated and committed again (`validation/regenerate_predictions.sh`,
plan step D17), and `run.sh --dry-run` is clean:

1. `compare.py` gave a **bit-exact** or **tolerance** row a verdict other than a match,
   the cause was found, and the fix (to the simulator, a probe, or the prediction) changes
   what that row predicts. Hardware must then confirm the new prediction.
2. Probes that tested nothing in the first rental: `FAILED TO LAUNCH`, `HOST ERROR`,
   `not run`, a stage that hit its time limit, or the instance was released early.
3. Row RT-11 was not captured: `out/blackwell/RT-11-NOT-CAPTURED.txt` is in the archive,
   or `rows/e2e/cute_dsl_tma/summary.txt` does not start with `RT-11 capture: COMPLETE`
   (`rows/status.tsv` then says `skipped` or `failed` for `e2e/cute_dsl_tma`). The tensor
   maps are what a decoder for CuTe DSL kernels with TMA is written against, the rest of
   the capture is what such a kernel is then replayed from, and nothing else produces
   either. `run.sh` exits 1 for this, so it is not found out at home.
4. `ncu` could not read counters (a container) and the timing fit needs them: rent a VM
   or bare metal. With the first rental's default this shows at preflight, in its first
   minutes, and not afterwards; it is a second rental only if the first went on with
   `--no-ncu` or `--keep-going` (`NCU-COUNTERS-NOT-READ.txt` in its archive).

What does not: a difference in an **informational** row; a mismatch that was explained
and fixed without changing a prediction that hardware has to confirm; a row that matched.

The second run is the same commands with what is not needed left out:

```bash
env/remote/run.sh --skip selftest                                   # everything again
env/remote/run.sh --skip selftest --skip build-timing --skip run-timing   # probes only changed
WARPLENS_BLACKWELL_RUN_ARGS="--only tma/*" env/remote/run.sh --skip selftest --skip build-timing --skip run-timing   # one family; no end-to-end, no rows
```

A third rental comes out of the same reserve, for the same reasons. When the 6 hours are
spent with mismatches still open, more is a new decision for the owner, not a default.

## 6. Getting results back (E22)

`run.sh` ends by printing three lines: the archive path, its size and its
SHA-256. The archive is `$WARPLENS_REMOTE_HOME/results/warplens-b200-<start>-<commit>.tar.gz`
with a `.sha256` next to it. It is rebuilt at the end of every `run.sh`, also
after a failure.

Contents:

| Path | What |
| --- | --- |
| `manifest.txt` | commit, branch, start and end times, host, the status of every stage; `ncu=ok` only when preflight read a counter, `ncu_required`, `ncu_counters`; `book` and `book_commit` (the clone used; the book itself is never in the archive) |
| `stages.tsv` | every attempt of every stage: status, exit code, start, end, seconds |
| `preflight-quick.txt`, `preflight.txt` | the two preflight reports |
| `resolved/` | what `setup.sh` installed: every conda package with its hash, `pip freeze`, `versions.txt`, `drift.txt` (differences from the pins). The archive's `versions.txt` ends with `ncu_counters=read` or `not-read (...)` |
| `pins.env` | what was asked for |
| `nvidia-smi.txt`, `nvidia-smi-q.txt` | the GPU as the driver describes it |
| `logs/<stage>.log` | latest output of each stage; earlier attempts are preserved in `logs/history/` |
| `out/blackwell/`, `out/timing/` | the suites' raw outputs (`WARPLENS_RESULTS_DIR`) |
| `out/blackwell/rows/` | the rows without a probe: `status.tsv`, the UE4M3 answer of ptxas (NUM-05), the driver call traces (RT-10), and in `e2e/cute_dsl_tma/` the capture of the CuTe DSL tutorial (RT-11): its output, the kernel parameters of its launch, tensor maps included, its code and PTX, the memory it reads and writes, `summary.txt` and `capture.sha256`. About 5 MB. Section 3 |
| `out/blackwell/RT-11-NOT-CAPTURED.txt` | only when the capture of row RT-11 was required and is missing: why. Section 3 |
| `out/predicted-toolkit/` | only with a fallback toolkit: the predictions made with the installed nvcc before the GPU stages, `DIFFERENCES.txt`, `toolkit.txt`. Section 2 |
| `PREDICTIONS-NOT-COMPARABLE.txt` | only with a fallback toolkit: why, and the two commands that compare with `out/predicted-toolkit/` |
| `NCU-COUNTERS-NOT-READ.txt` | only when preflight read no counter: whether the run was told to do without (`--no-ncu`), did not need them (no timing suite) or required them and went on with `--keep-going` |
| `suite-dirs/` | result directories a suite created inside its own directory |
| `SHA256SUMS` | a checksum for every file above |

The archive includes executables, full tensor buffers and a source snapshot as
well as text logs. The 17-case CPU-only end-to-end capture occupied 73 MB before
compression; a full rental adds probe artifacts, timing results and the frozen
source tree. Check the reported archive size and allow space for two copies.
Transfer it with `scp` or `rsync` over the existing SSH connection.

## 7. When preflight fails

| Check | Meaning | What to do |
| --- | --- | --- |
| `nvidia-smi` FAIL | no driver visible | A container started without the GPU. Nothing to fix from inside: release it. |
| `gpu-identity` FAIL | not compute capability 10.0 | **Release the instance.** The message names the card. Nothing else is worth running. |
| `driver` FAIL | driver older than CUDA 12.8 | Cannot run `sm_100a`. Release, or pick another image with a newer driver. |
| `driver` WARN | driver older than the pinned toolkit | `setup.sh` matches the toolkit to it by default, and the results are then flagged as not comparable with the predictions. Decide between that and `--toolkit pinned`: section 2, "Driver older than the toolkit". |
| `predictions` WARN | the nvcc in the env is not the one the predictions were made with (`BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE`), or the driver refuses its PTX and machine code will run | The first is the fallback of the toolkit policy (section 2): go on; `run.sh` makes the predictions again with this nvcc in its stage `predict-toolkit`. The second needs nothing. `run.sh --strict-toolkit` stops instead of going on. |
| `torch` WARN | no PyTorch env, or it does not see the GPU | A full run fails its native E2E prerequisite check; a skipped Triton step also fails collection. Retry `env/remote/setup.sh` or fix GPU visibility. With a 12.x driver the pinned PyTorch cannot be installed; choose a compatible driver or deliberately use `--no-e2e`. |
| `gpu-memory` / `gpu-idle` WARN | something else is on the GPU | If it is yours, stop it. If not, the card is shared: functional results are fine, timing is not. Ask for a dedicated GPU or note it. |
| `disk` FAIL | under 18 GB (2 GB once installed) | `export WARPLENS_REMOTE_HOME=/path/on/the/big/disk` and start again. `setup.sh --no-torch` needs 12 GB. |
| `cpu` / `ram` WARN | small host | Builds are slower; nothing else. |
| `clock` WARN | clock wrong | Downloads may fail on certificate dates and the manifest's times are off. `sudo timedatectl set-ntp true` if you can; otherwise write down the true time next to the archive. |
| `env` FAIL | `setup.sh` has not run | Run `env/remote/setup.sh`, or use `run.sh`, which does. |
| `compile` FAIL | nvcc cannot build for `sm_100a` | The env is broken: `env/remote/setup.sh --force`. |
| `driver-api` FAIL | the driver API disagrees with `nvidia-smi` | Several GPUs in a different order: `export CUDA_VISIBLE_DEVICES=<index of the B200>`. |
| `run-trivial` FAIL | `sm_100a` machine code does not run | With `gpu-identity` PASS this should not happen. Read the error; try `env/remote/setup.sh --toolkit <the driver's CUDA Version> --force`. If it persists the machine is unusable. |
| `ptx-jit` WARN | the driver refuses the toolkit's PTX | `--native` covers the probes only. Full end-to-end runs also fail `e2e-ptx`; use a driver-compatible toolkit (`--toolkit auto`) or deliberately select `--no-e2e`. |
| `rt11-tracer` WARN | no gdb, a gdb without Python, or a machine that does not let gdb trace a process | Matrix row RT-11 cannot be captured without it, and on a B200 the stage `run-blackwell` then fails at its end (section 3, "The rows without a probe"). Fix it now: `apt-get install -y gdb` as root, or `env/remote/setup.sh` again, or `WARPLENS_GDB`; in a container, `--cap-add=SYS_PTRACE`. The probes do not need it. |
| `native-e2e` FAIL | the PyTorch env does not see the GPU, or there is no clone of the book | The message says which. Both: `env/remote/setup.sh` again (it installs the one and fetches the other; section 3, "PyTorch and the book"), or through `run.sh`: `env/remote/run.sh --redo setup`. If the book cannot be fetched from this machine: the fallback there, by hand, and `export WARPLENS_BOOK`. Then `env/remote/run.sh` again. For probes only, set `WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e`. |
| `book` FAIL | `setup.sh`'s clone in `$WARPLENS_REMOTE_HOME/book` is not at the pinned commit | Something changed it. `env/remote/run.sh --redo setup` fetches the pinned commit again. |
| `book` WARN | the clone you named (`WARPLENS_BOOK`, `--book`) is at another commit, or at none that can be read | Row RT-08 then runs other kernels than the prediction was made for. Unset `WARPLENS_BOOK` to use `setup.sh`'s clone, or note it next to the archive. |
| `tcgen05` / `tensor-map` / `tma` FAIL | a Blackwell feature is not exposed | On a real B200 these must pass. Keep the report (it is in the archive) and stop: this is a finding about the machine or a bug in the preflight kernel, not something to debug on the clock. `--keep-going` runs the suites anyway if the failure is confined to one feature. |
| `nvrtc` WARN | NVRTC cannot compile for `sm_100a` | TIRx kernels will not build. Expected only with a 12.x toolkit. |
| `ncu` FAIL, `ERR_NVGPUCTRPERM` | no permission to read counters. **The run stops here**: a full run requires a counter read, because the timing model is held against counters | In order: run everything as root (`sudo -i`, then the commands of section 3). On a VM as root, if it still fails: `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` needs the module reloaded, which is rarely possible with the GPU in use. In a container: only the provider can grant it (`--cap-add=SYS_ADMIN`). If none works, decide: release the machine and rent one that grants it (section 2, "Before paying"), or take everything but the counters with `env/remote/run.sh --no-ncu`. |
| `ncu` FAIL, other | no ncu was found, ncu ran but gave no counter, or the trivial kernel did not run so there was nothing to profile | The report shows which, with ncu's last lines. No ncu: `env/remote/run.sh --redo setup` without `--no-ncu`. An ncu too old to know the B200 says so in those lines; `setup.sh` installs one that matches the toolkit. Nothing to profile: fix `run-trivial` first. Otherwise as above: `--no-ncu` to go on without. |
| `ncu` WARN | the same outcomes, in a run that was told to do without counters (`--no-ncu`) or has no timing suite (`--skip run-timing`) | Nothing: the run goes on. The archive says that no counter was read (`NCU-COUNTERS-NOT-READ.txt`, `ncu=unavailable` and `ncu_counters=` in `manifest.txt`, `ncu_counters=` at the end of `resolved/versions.txt`). |

**Counters are required by default** (decided by the owner on 2026-11-24). What the options
do about it:

| Command | A counter could not be read |
| --- | --- |
| `env/remote/run.sh` | preflight FAILs and the run stops before the builds and the GPU stages; the archive of what exists is written, exit status 3 |
| `env/remote/run.sh --no-ncu` | a WARN; everything else runs; `setup.sh` installs no ncu (an env that already has one is kept as it is, not reinstalled); the archive says that no counter was read |
| `env/remote/run.sh --keep-going` | as for every failed preflight: the suites run, the preflight stage stays `failed`, exit status 1, and the archive says that counters were required and not read |
| `env/remote/run.sh --skip build-timing --skip run-timing` | a WARN: without the timing suite nothing reads counters |
| `env/remote/preflight.sh` / `preflight.sh --no-ncu` | FAIL / WARN, as the first two rows |

Only a PASS of the `ncu` check counts as a counter read (`WARPLENS_NCU=ok` for the suites,
`ncu=ok` in the manifest): it means ncu printed a value for the counter of a kernel that
ran. A check that was skipped, or never reached, is `unavailable`.

A stage of `run.sh` that fails prints the end of its log and the run goes on.
Archive or checksum failures return a nonzero status; preserve the raw run directory
and retry `--collect-only` after fixing the storage problem. Fix, then run `env/remote/run.sh` again.

`run-blackwell` fails for missing programs or artifacts, launch or host errors,
unexpected process failures or timeouts, and incomplete or failed E2E captures.
Probe faults and hangs remain valid observations only when the manifest permits
them. Differences in the output of a completed probe remain results to assess
with `compare.py`; they do not by themselves fail collection.
`validation/blackwell/README.md`, "When probes do not launch", says what to do.
In the timing suite the same thing is `LAUNCH FAILED` in `status.txt`.

## 8. The last minutes before releasing the instance

The machine's disk is gone when it is released. In this order:

1. `run.sh` has printed `archive:`, `size:` and `sha256:`. If a stage is still
   running and time is up, press Ctrl-C once: `run.sh` stops the stage, packs
   what exists and prints the same three lines. `env/remote/run.sh --collect-only`
   packs again without running anything.
2. `run.sh` did not end with the banner `MATRIX ROW RT-11 WAS NOT CAPTURED`, and
   `head -1 $WARPLENS_REMOTE_HOME/run/out/blackwell/rows/e2e/cute_dsl_tma/summary.txt` says
   `RT-11 capture: COMPLETE`. If not, section 3, "The rows without a probe", has the
   commands; they take a minute, and a missing capture is a reason for a second rental
   (section 5). Then `env/remote/run.sh --collect-only`.
3. Copy the archive and its `.sha256` to your machine. Run `sha256sum -c` **there**
   and compare the value with the one `run.sh` printed.
4. Unpack it locally and check `manifest.txt`: the commit is the one you wrote
   down, and every stage has the status you expect. `sha256sum -c SHA256SUMS`
   inside the unpacked directory checks every file.
5. Copy the raw outputs a second time by a second route, in case the archive
   step missed something: `rsync -a <user>@<host>:warplens-remote/run/ ./run-raw/`
   and, if the suites wrote inside the checkout, `rsync -a <user>@<host>:warplens/validation/ ./validation-raw/`.
6. Put a second copy of the archive somewhere that is not this laptop.
7. Save `~/run.log` (the console output) too.
8. Only then release the instance, and check in the provider's console that
   billing has stopped.

## 9. What was and was not tried

Measured on the development machine, into throwaway directories that were removed afterwards:

- `setup.sh` from nothing, as a normal user without conda on `PATH`: pinned
  toolkit 13.5 (`--toolkit pinned`), the driver-matched 13.0 (the default there,
  since that driver supports CUDA 13.0), and 12.8. In the 13.5 env every pin was
  met exactly. In the 13.5 and 13.0 envs the repository's whole test suite passed
  (the tests that need the CUTLASS programs or a clone of the book were skipped,
  as they are without those).
- Without `git` on `PATH` (it is installed into the env) and without any
  download tool (a clear error).
- `preflight.sh` on the RTX 3080 Ti: reports that it is not a B200 and fails.
  With `PREFLIGHT_EXPECT_CAP=8.6`, the compile, driver API, trivial-kernel and
  PTX checks ran for real against that GPU, with both toolkits; the `tcgen05`,
  tensor-map and TMA checks failed as they must on Ampere; `ncu` was refused
  with `ERR_NVGPUCTRPERM`, then a WARN with the remedies (since 2026-11-24 a
  FAIL; below).
- The two preflight kernels (`tcgen05.alloc`/`dealloc`, the TMA load) run
  correctly in the simulator, from the checked-in PTX.
- `run.sh` end to end against the 13.0 env, with copies of the two probe suites
  as they stood that day: both built; the timing suite ran (every benchmark
  failing on Ampere) and its outputs were archived; resuming skipped finished
  stages.
- Again on 2026-11-23 with the finished suites (101 probe runs, 14 benchmarks,
  the PyTorch env): `setup.sh` from nothing took 79 s for the main env and 144 s
  for the PyTorch env; the first attempt at the latter failed with "No space
  left on device" because pip unpacked into a small `/tmp`, which is why
  `setup.sh` now keeps its temporary files under `WARPLENS_REMOTE_HOME`, and a
  second `setup.sh` retried just that part. `run.sh --keep-going` with
  `PREFLIGHT_EXPECT_CAP=8.6` took 13 min 52 s: the self-test 10 min 31 s (1513
  tests), the builds 8 s and 62 s, the probe run 2 min (every Blackwell probe
  reported as failed to launch, the stage failing for it), the timing run 4 s.
  The Triton step, the TIRx script and the CuTe DSL script ran on the GPU; the
  book's kernels were compiled and reached the driver, then failed as they must
  on Ampere. Ctrl-C during the self-test stopped the stage and wrote the
  archive.
- What the fallback toolkit does to the predictions: with nvcc 13.0.88 the
  simulator printed the committed predictions for all 98 predicted probe runs
  (after a fix to the probes; `validation/blackwell/README.md`, finding 16). So
  for the probes a fallback toolkit is flagged but, as far as the simulator can
  tell, harmless. On the rented machine the self-test repeats this check with
  whatever nvcc is installed (`tests/test_predict_path.py`): its per-probe
  failures name the predictions that do not hold for that compiler. Instruction
  counts do differ between the two compilers (one or two per kernel in the
  guide's examples), so the timing predictions are compiler-specific.
- The self-test with that fallback toolkit is not all green, and need not be:
  expect `tests/test_predict_path.py::test_the_compiler_is_the_one_the_committed_predictions_name`
  (it says which nvcc is installed), four `tests/test_clc.py::test_the_kernels_assemble`
  cases (their PTX says `.version 9.5`, which ptxas 13.0 refuses) and three
  `tests/test_guide.py` examples (the guide prints instruction counts of nvcc
  13.5 code). In the rehearsal there were 26 failures: those 8, the 16 probe
  predictions that finding 16 has since fixed, and 2 caused by the rehearsal's
  own `PREFLIGHT_EXPECT_CAP`. Any other failure on the day is news.
- On 2026-11-24, the additions of that day (toolkit policy, rows without a probe):
  - `env/remote/predict_toolkit.sh --build` with nvcc 13.5.37, the only `sm_100a`
    compiler installed: 9 min 40 s in all on 20 cores, of which 65 s for the 108 probe
    predictions. As it must with the predictions' own compiler, it reproduced every
    committed probe prediction and every timing prediction (`DIFFERENCES.txt` empty).
    That exercises the script, not the fallback: with another nvcc it has not run.
  - The stage `predict-toolkit` of `run.sh`, with stand-in suites only
    (`tests/test_remote_env.py`): it runs before the GPU stages when
    `resolved/versions.txt` says `not-comparable`, its directory and the names that
    differ are in the archive, a failure does not stop the GPU stages, and it is skipped
    with the pinned toolkit. Not run inside a whole `run.sh` with a real fallback env.
  - `validation/blackwell/run_rows.py` on the RTX 3080 Ti (driver 580.173.02): NUM-05
    `ok` (ptxas 13.5 accepts the eight control conversions and none for UE4M3); the
    CuTe DSL trace `ok` in 5 s (26 distinct driver calls, one `cuLaunchKernelEx`, its three
    parameters read); the TIRx trace logged 33 distinct calls and ended as `failed`,
    because this driver refuses NVRTC 13.5's PTX at `cuModuleLoadData`, as the untraced
    script does here; RT-11 `skipped (needs sm_100 ...)`.
  - The capture of launch parameters, with a CUDA C++ program for `sm_86` whose kernel
    takes a 128-byte, 64-byte-aligned struct by value (`cuLaunchKernel`, the other entry
    point): the driver reported sizes 4, 128, 8 and the 128 bytes saved were the bytes
    passed. With the system gdb 15.1 and with conda-forge's gdb 18.1.
  - `shellcheck` 0.11 on the five scripts, `predict_toolkit.sh` included: clean.
- On 2026-11-24, the book fetch and the required counters, as a normal user on the RTX
  3080 Ti (driver 580.173.02), into a directory of its own that was removed afterwards:
  - `setup.sh --no-torch` from nothing, 2 min 19 s: it cloned the book from github.com
    into `$WARPLENS_REMOTE_HOME/book` at the pinned commit (11 MB) and recorded
    `book_commit=` in `resolved/versions.txt`. A second `setup.sh`, with `--no-ncu` added,
    took 4 s: `the book: present`, and the env that already had ncu was kept, not
    reinstalled.
  - A fetch from a repository github.com does not have: `failed`, at once and without a
    question (git asks for a user name there unless told not to), nothing left behind.
  - `preflight.sh --e2e` with `PREFLIGHT_EXPECT_CAP=8.6`: `PASS book` for that clone;
    `FAIL ncu` with `ERR_NVGPUCTRPERM` and the remedies (ncu 2025.3.1, the release for
    toolkit 13.0); with `--no-ncu` the same as `WARN ncu`.
  - `run.sh --no-torch --skip selftest --skip predict-toolkit` with
    `PREFLIGHT_EXPECT_CAP=8.6` and `--no-e2e`: took the clone for `WARPLENS_BOOK` by itself,
    stopped after preflight with exit status 3 in 12 s and said why (no counter was read;
    on this GPU the three Blackwell checks fail too), and wrote an archive with
    `NCU-COUNTERS-NOT-READ.txt`, `ncu=unavailable`, `ncu_required=1`, `book=` and
    `book_commit=` in its manifest and no file of the book.
  - With stand-ins only (`tests/test_remote_regressions.py`): the book over a clone at
    another commit, a commit that is not the book, a run that goes on with `--no-ncu` or
    `--keep-going`, and every `PASS ncu` path.

Later on 2026-11-24, `run.sh --keep-going` with `PREFLIGHT_EXPECT_CAP=8.6` from an empty
home, on the tree with the pre-rental gap work merged (118 probe runs), 32 min in all,
exit status 1 as it must be on Ampere:

- `setup.sh` took 208 s and fell back to nvcc 13.0.88 for this driver, so this was the
  first whole run of the fallback path: `predict-toolkit` ran by itself before the GPU
  stages (350 s) and wrote `out/predicted-toolkit/` with `DIFFERENCES.txt` into the archive.
- With nvcc 13.0.88 two probe predictions differ and thirteen timing files do.
  `formats/e8m0_ties` and `formats/special_inputs` end with `call.uni __assertfail`:
  13.0's `__nv_cvt_float_to_e8m0` is an assertion in device code, so with a 13.0 toolkit
  the UE8M0 encodings of rows NUM-04 and NUM-06 cannot be measured on the GPU either. A
  13.0 driver on the day costs those lines; nothing else among the probes.
- The self-test (2926 passed, 14 failed, 4 errors, 17 min) failed only where the compiler
  or the dry run's own variable decides: the installed-compiler test, the two probes above,
  four `tests/test_clc.py::test_the_kernels_assemble`, four `tests/test_guide.py` blocks
  (instruction counts), four `tests/test_atomics.py::test_compiled_cuda_atomic_add_runs_without_a_gpu`
  (nvcc 13.0 spells `atom` otherwise than the test expects), and three preflight tests that
  read `PREFLIGHT_EXPECT_CAP`. With the pinned 13.5 env the same tree passes whole
  (2969 passed, 8 skipped).
- The probe run: 86 probes failed to launch and 28 had a host error, as on any GPU that is
  not a B200; the rows without a probe ran (`ue4m3_ptxas` ok, both driver traces ok, RT-11
  skipped for want of `sm_100`); the stage and `run.sh` reported the failure.
- In the simulator, `e2e/run_e2e.py --mode sim` exits 0 with all eight steps `ok`, and
  exits 1 with `e2e/book_hgemm` as `exit 1` when the book's runner is made to fail.
- On 2026-11-24, the capture of row RT-11 (code, memory before and after; section 3), all
  on the RTX 3080 Ti, driver 580.173.02, each with the system gdb 15.1 and with
  conda-forge's gdb 18.1:
  - `cute_dsl_tma.py --script examples/06_cute_dsl/saxpy.py`, the whole path: `capture:
    complete`. The DSL hands the driver a fat binary at `cuLibraryLoadData` (3584 bytes),
    not PTX; the cubin of the DSL's own dump is inside it byte for byte, and the PTX of
    that dump is the PTX the frontend capture of the same script saves. The two 1024-byte
    allocations were followed from `cuMemAlloc_v2`, the driver's `cuMemGetAddressRange`
    agrees, and the dumps are the script's inputs before the kernel and its result after,
    bit for bit (`tests/test_launch_capture.py`).
  - A CUDA C++ kernel that takes three 128-byte, 64-byte-aligned structs by value
    (`tests/data/launch_capture_descriptors.cu`): the address at offset 0 and the one at
    offset 3 were found, the one stored shifted by 4 bits was not, and its memory was in
    the capture all the same (every live allocation is read when a parameter is that
    large); the after dump of the output is the expected sum.
  - A CuTe DSL kernel over PyTorch tensors (`tests/data/cute_dsl_torch_saxpy.py`, torch
    2.15.0+cu132 in a venv made as `install_torch_env` makes one), traced as the tutorial
    is: both tensors inside one 2 MiB allocation of PyTorch's allocator, followed through
    `cudaMalloc`; PyTorch's own four launches listed, their memory not read, their three
    modules (about 1 MB each) not kept; 4.1 MB in all, 7 s.
  - `fp16_gemm_0.py` itself with `--force` and `CUTE_DSL_ARCH=sm_100a`, so that the DSL
    compiles the real kernel for a GPU that is not there: the capture run saved its fat
    binary (35664 bytes), the DSL's PTX (26855 bytes, six parameters, two of them
    `.align 64`) and cubin (inside the fat binary), the kernel's name, and the one 2 MiB
    allocation that holds the three tensors at `--mnk 256,512,512`. It ends at
    `cuLibraryGetKernel` with error 209, no binary for this GPU, before any launch.
    Without `CUTE_DSL_ARCH` the DSL refuses `tcgen05` for `sm_86` before it compiles.
  - What went wrong on the way, and is why the tracer is as it is: a call made in the
    stopped program with its other threads held (`cuKernelGetParamInfo` at a launch by
    PyTorch) never returned. The capture's calls now run with the other threads running;
    gdb 15's time limit on such calls was seen to end one after its limit and hand the
    program back; `capture.log` names the call. And gdb 18.1 would not let the program go
    on once the CuTe DSL's runtime library was unloaded at exit with a breakpoint of the
    tracer still in it (where the launch returned to); such breakpoints are now removed
    as soon as the call has returned.
  - `run_rows.py`: RT-11 `skipped (needs sm_100 ...)` and exit status 0; with
    `--rt11 required`, exit status 1, the banner and `RT-11-NOT-CAPTURED.txt`.

Not tried, because there was no way to here:

- Anything on a B200: every PASS path of the `tcgen05`, tensor-map, TMA and
  `ncu` checks. The simulator says the kernels are right; hardware has not.
- Root, in a container or not: a real `apt-get`, a real root shell, and `ncu`
  succeeding. No container runtime was usable on the development machine
  (docker needs its group, podman is not installed, and Ubuntu 24.04 denies
  unprivileged user namespaces). What stands in: `tests/test_remote_setup_paths.py`
  runs `setup.sh` up to the download in a bare `PATH`, as a normal user and as
  root under `fakeroot` with a stand-in `apt-get` (the decisions, not the
  privileges), and `shellcheck` 0.11 is clean on every script here.
- `--toolkit pinned` on a driver older than the toolkit, as a whole run.
- Ubuntu 22.04 (the development machine is 24.04), any provider's image, a slow
  network, Arm.
- RT-11 itself: the launch of `fp16_gemm_0.py` has not happened under the tracer
  anywhere, because it needs a B200. Everything up to it has, on the RTX 3080 Ti (the
  list above): the tutorial compiled for `sm_100a`, its module and PTX saved, its
  allocation followed. First on the B200: the launch's parameters (that the tensor maps
  arrive as two 128-byte, 64-byte-aligned parameters is what the kernel's PTX says; where
  in them the address is, is not known), the scan of those parameters for addresses, the
  reads of device memory at that launch, `cuCtxSynchronize` after a `tcgen05` kernel, and
  the tutorial passing at 256 x 512 x 512. If the tensor maps are not parameters,
  `calls.txt` of the same trace shows what the DSL called instead.
- A gdb older than 15 (Ubuntu 22.04 has 12.1) for the capture of RT-11. It has no time
  limit for a call made in the program, so a call that does not return costs the capture
  run its whole timeout (15 min); what was saved before it is kept. Nothing else in the
  tracer is known to need a newer gdb, and none was tried.
- gdb inside a provider's container. Starting a child under gdb needs `ptrace`, which
  Docker's default seccomp profile allows on current kernels; a stricter profile makes
  the two tracing rows `skipped (... ptrace is not permitted)`.
- `setup.sh`'s new gdb step in a real run: the three branches (a gdb on `PATH`,
  `apt-get install gdb` as root, conda-forge's gdb into `gdb-env`) were not exercised
  through `setup.sh`; conda-forge's gdb was installed by hand with the same package
  spec and ran the tracer.
- `PASS ncu` for real: it needs root, here as on the rented machine.
- The book fetched by a git that `setup.sh` put into the env (a machine without git):
  `git_run` in `lib.sh` falls back to it, and no machine without git was at hand.
- The by-hand fetch of the book of section 3 on this machine (the commands are CI's;
  `setup.sh` runs the same three git commands).
- `predict_toolkit.sh` with a toolkit between 13.1 and 13.4 (13.0.88 was run, above).
- A driver newer than the pinned toolkit, and a 12.x driver (only a 12.8
  *toolkit* on a 13.0 driver).
