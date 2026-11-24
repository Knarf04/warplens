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
| `setup.sh` | Installs the env from the pins. No root, no conda, no system toolkit needed. |
| `preflight.sh` | Is this machine a real B200 and does everything on it work? PASS / WARN / FAIL per check. |
| `run.sh` | Preflight, setup, build and run both probe suites, pack one archive. Resumable. |
| `gpu_probe.py`, `check_env.py`, `kernels/` | What preflight runs on the GPU. |

## 1. Before renting

Do these at home; none needs the B200.

- [ ] The commit to test contains `validation/blackwell/`, `validation/blackwell_timing/`
      and `env/remote/`. `env/remote/run.sh --dry-run` must show no stage as `MISSING`.
- [ ] The predicted outputs are committed (plan step D17).
- [ ] `python -m pytest -q tests/test_remote_env.py` passes.
- [ ] A dry run of preflight on the development GPU fails for the right reason:
      `WARPLENS_ENV_RUN="conda run -n warplens" env/remote/preflight.sh` must say
      `FAIL gpu-identity ... not a B200`.
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
  nothing you do inside helps. The run completes either way; without counters
  the `ncu` comparison in the timing suite (plan step C14) is skipped.
- **Driver.** Ask for, or pick the image with, the newest driver. What matters is
  the "CUDA Version" in the header of `nvidia-smi`:
  - 13.5 or newer: everything runs as pinned.
  - 13.0 to 13.4: `setup.sh` installs that toolkit release in place of 13.5
    (measured on a 13.0 driver: PTX from nvcc 13.5 is refused with
    `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, machine code from it runs, and with
    toolkit 13.0 installed both run).
  - 12.8 or 12.9: the CUDA C++ probes can run with toolkit 12.x, but TIRx and the
    CuTe DSL will not (measured with toolkit 12.8: their tests fail; they need
    `libcudart.so.13`). Avoid if there is a choice.
  - below 12.8: unusable; preflight fails.
- **Disk:** 12 GB free where `WARPLENS_REMOTE_HOME` points (measured: the env is
  4.8 GB and the package cache 1 GB). **Network:** outbound HTTPS to github.com,
  conda.anaconda.org and pypi.org.
- A shared card (another tenant on the same GPU) makes the timing numbers
  worthless. Preflight warns when it sees other processes or load, but from
  inside a container it cannot see other tenants. Rent a dedicated GPU.

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
export WARPLENS_REMOTE_HOME=$HOME/warplens-remote     # 12 GB; point it at the big disk
env/remote/preflight.sh --quick              # 2 seconds. FAIL gpu-identity: release the machine now
env/remote/run.sh --dry-run                  # the plan; no stage may say MISSING
tmux new -s run
env/remote/run.sh 2>&1 | tee ~/run.log       # everything; prints the archive path and checksum at the end
```

- No `git` on the machine: `setup.sh` installs it with `apt-get` when it is root
  (not exercised) or puts conda-forge's git into the env (measured). To get the
  checkout there without git, copy the tree instead:
  `rsync -a --exclude __pycache__ ./ <user>@<host>:warplens/` (include `.git`;
  the commit is read from it without a git program).
- If you are not root and `sudo` works, become root first (`sudo -i`) and do
  everything as root, so that `ncu` may read counters. Decide this before the
  first command: the env lives under `$HOME` of whoever runs `setup.sh`.
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

The pinned setup does **not** install PyTorch or fetch the book. A full run now
checks both before starting the suites: provide a compatible **CUDA-enabled**
PyTorch installation in the remote environment and set `WARPLENS_BOOK` to the
book clone (or pass `--book` through `WARPLENS_BLACKWELL_RUN_ARGS`). A CPU-only
PyTorch wheel does not satisfy the native Triton path. No PyTorch/toolkit pair
has been validated for this rental yet; resolve that environment before renting.
For a deliberately narrower run of the hardware probes without end-to-end programs:

```bash
WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e env/remote/run.sh
```

A standalone `preflight.sh` checks the GPU and probes; add `--e2e` to check these
full-suite prerequisites. `run.sh` selects that check automatically unless the
Blackwell suite is skipped or restricted with `--no-e2e` / `--only`.

## 4. What each stage takes

`env/remote/run.sh --dry-run` prints this table for the stages still to run.

| Stage | Minutes | Basis |
| --- | --- | --- |
| log in, upload, clone | 5 to 10 | guess |
| `preflight-quick` | under 1 | measured: 2 s |
| `setup` | 2 to 8 | measured: 57 to 64 s for a fresh install (5.8 GB on disk) on a fast link; the upper end is a guess for a slow link or disk |
| `preflight` | 1 to 2 | measured: 5 s, but there the Blackwell kernels fail at load; on a B200 they run, and `ncu` attaches |
| `selftest` (the repository's tests, in the simulator) | 1 to 3 | measured: 50 s on 20 cores |
| `build-blackwell` | 1 to 6 | measured: 8 s on 20 cores (67 CPU-seconds) for an unfinished copy of the suite, 21 probes; upper end is a guess for the finished suite |
| `build-timing` | 1 to 4 | measured: 41 s for an unfinished copy of the suite |
| `run-blackwell` | 3 to 20 | guess. Kernels are tiny; the time is process start-up and the probes that wait out a timeout on purpose |
| `run-timing` | 3 to 20 | guess. Under `ncu` every kernel is replayed per counter group |
| `collect` | under 1 | measured |
| copy back, verify, second copy | 5 | guess |
| **Total** | **25 to 80** | the two run stages are the unknown |

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
| Everything works first time | 0.5 to 1.3 | $4 to $11 |
| Budget, with one retry of a stage | 2 | $16 |
| Provider bills whole hours | 1 or 2 | $8 or $16 |
| Only an 8-GPU node is on offer | 2 x 8 GPUs | about $130 |

Plan step F24 budgets a second, shorter rental; with `--skip selftest` and the
same commands it costs the same or less.

## 6. Getting results back (E22)

`run.sh` ends by printing three lines: the archive path, its size and its
SHA-256. The archive is `$WARPLENS_REMOTE_HOME/results/warplens-b200-<start>-<commit>.tar.gz`
with a `.sha256` next to it. It is rebuilt at the end of every `run.sh`, also
after a failure.

Contents:

| Path | What |
| --- | --- |
| `manifest.txt` | commit, branch, start and end times, host, the status of every stage |
| `stages.tsv` | every attempt of every stage: status, exit code, start, end, seconds |
| `preflight-quick.txt`, `preflight.txt` | the two preflight reports |
| `resolved/` | what `setup.sh` installed: every conda package with its hash, `pip freeze`, `versions.txt`, `drift.txt` (differences from the pins) |
| `pins.env` | what was asked for |
| `nvidia-smi.txt`, `nvidia-smi-q.txt` | the GPU as the driver describes it |
| `logs/<stage>.log` | latest output of each stage; earlier attempts are preserved in `logs/history/` |
| `out/blackwell/`, `out/timing/` | the suites' raw outputs (`WARPLENS_RESULTS_DIR`) |
| `suite-dirs/` | result directories a suite created inside its own directory |
| `SHA256SUMS` | a checksum for every file above |

Expected size: small. Measured 30 kB for a run in which every benchmark failed
at once; a real run is text output of a few hundred probes, so a guess is under
10 MB. `scp` or `rsync` over the ssh connection you already have is enough; no
object storage is needed.

## 7. When preflight fails

| Check | Meaning | What to do |
| --- | --- | --- |
| `nvidia-smi` FAIL | no driver visible | A container started without the GPU. Nothing to fix from inside: release it. |
| `gpu-identity` FAIL | not compute capability 10.0 | **Release the instance.** The message names the card. Nothing else is worth running. |
| `driver` FAIL | driver older than CUDA 12.8 | Cannot run `sm_100a`. Release, or pick another image with a newer driver. |
| `driver` WARN | driver older than the pinned toolkit | Nothing to do: `setup.sh` matches the toolkit to it. Note it; results come from that toolkit, recorded in `resolved/`. |
| `gpu-memory` / `gpu-idle` WARN | something else is on the GPU | If it is yours, stop it. If not, the card is shared: functional results are fine, timing is not. Ask for a dedicated GPU or note it. |
| `disk` FAIL | under 12 GB (2 GB once installed) | `export WARPLENS_REMOTE_HOME=/path/on/the/big/disk` and start again. |
| `cpu` / `ram` WARN | small host | Builds are slower; nothing else. |
| `clock` WARN | clock wrong | Downloads may fail on certificate dates and the manifest's times are off. `sudo timedatectl set-ntp true` if you can; otherwise write down the true time next to the archive. |
| `env` FAIL | `setup.sh` has not run | Run `env/remote/setup.sh`, or use `run.sh`, which does. |
| `compile` FAIL | nvcc cannot build for `sm_100a` | The env is broken: `env/remote/setup.sh --force`. |
| `driver-api` FAIL | the driver API disagrees with `nvidia-smi` | Several GPUs in a different order: `export CUDA_VISIBLE_DEVICES=<index of the B200>`. |
| `run-trivial` FAIL | `sm_100a` machine code does not run | With `gpu-identity` PASS this should not happen. Read the error; try `env/remote/setup.sh --toolkit <the driver's CUDA Version> --force`. If it persists the machine is unusable. |
| `ptx-jit` WARN | the driver refuses the toolkit's PTX | `--native` covers the probes only. Full end-to-end runs also fail `e2e-ptx`; use a driver-compatible toolkit (`--toolkit auto`) or deliberately select `--no-e2e`. |
| `native-e2e` FAIL | native Triton or book prerequisites are unavailable | Provide CUDA-enabled PyTorch and the book clone, then retry. For probes only, set `WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e`. |
| `tcgen05` / `tensor-map` / `tma` FAIL | a Blackwell feature is not exposed | On a real B200 these must pass. Keep the report (it is in the archive) and stop: this is a finding about the machine or a bug in the preflight kernel, not something to debug on the clock. `--keep-going` runs the suites anyway if the failure is confined to one feature. |
| `nvrtc` WARN | NVRTC cannot compile for `sm_100a` | TIRx kernels will not build. Expected only with a 12.x toolkit. |
| `ncu` WARN, `ERR_NVGPUCTRPERM` | no permission to read counters | In order: run everything as root (`sudo -i`, then the commands of section 3). On a VM as root, if it still fails: `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` needs the module reloaded, which is rarely possible with the GPU in use. In a container: only the provider can grant it (`--cap-add=SYS_ADMIN`). If none works, go on: only the counter comparison is lost. |
| `ncu` WARN, other | ncu ran but gave no counter | The report shows its last lines. An ncu too old to know the B200 says so there; `setup.sh` installs one that matches the toolkit unless `--no-ncu` was given. |

A stage of `run.sh` that fails prints the end of its log and the run goes on.
Archive or checksum failures return a nonzero status; preserve the raw run directory
and retry `--collect-only` after fixing the storage problem. Fix, then run `env/remote/run.sh` again.

## 8. The last minutes before releasing the instance

The machine's disk is gone when it is released. In this order:

1. `run.sh` has printed `archive:`, `size:` and `sha256:`. If a stage is still
   running and time is up, press Ctrl-C once: `run.sh` stops the stage, packs
   what exists and prints the same three lines. `env/remote/run.sh --collect-only`
   packs again without running anything.
2. Copy the archive and its `.sha256` to your machine. Run `sha256sum -c` **there**
   and compare the value with the one `run.sh` printed.
3. Unpack it locally and check `manifest.txt`: the commit is the one you wrote
   down, and every stage has the status you expect. `sha256sum -c SHA256SUMS`
   inside the unpacked directory checks every file.
4. Copy the raw outputs a second time by a second route, in case the archive
   step missed something: `rsync -a <user>@<host>:warplens-remote/run/ ./run-raw/`
   and, if the suites wrote inside the checkout, `rsync -a <user>@<host>:warplens/validation/ ./validation-raw/`.
5. Put a second copy of the archive somewhere that is not this laptop.
6. Save `~/run.log` (the console output) too.
7. Only then release the instance, and check in the provider's console that
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
  with `ERR_NVGPUCTRPERM`, as a WARN with the remedies.
- The two preflight kernels (`tcgen05.alloc`/`dealloc`, the TMA load) run
  correctly in the simulator, from the checked-in PTX.
- `run.sh` end to end against the 13.0 env, with copies of the two probe suites
  as they stood that day: both built; the timing suite ran (every benchmark
  failing on Ampere) and its outputs were archived; resuming skipped finished
  stages.

Not tried, because there was no way to here:

- Anything on a B200: every PASS path of the `tcgen05`, tensor-map, TMA and
  `ncu` checks. The simulator says the kernels are right; hardware has not.
- Root, in a container or not: the `apt-get` branches of `setup.sh`, and `ncu`
  succeeding. No container runtime was usable on the development machine.
- Ubuntu 22.04 (the development machine is 24.04), any provider's image, a slow
  network, Arm.
- A driver newer than the pinned toolkit, and a 12.x driver (only a 12.8
  *toolkit* on a 13.0 driver).
