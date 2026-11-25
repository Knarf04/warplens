#!/usr/bin/env bash
# Fetch CUTLASS at a pinned commit and compile its first four Blackwell tutorials to PTX.
# CUTLASS sources and the PTX generated from them are not checked in.
#
#   conda run -n warplens examples/03_cutlass_mma/build.sh           # PTX only
#   conda run -n warplens examples/03_cutlass_mma/build.sh --programs # also the host programs
#
# With --programs each tutorial is also built as an ordinary executable in bin/, for
#   warplens run examples/03_cutlass_mma/bin/tutorial_01
# The flags that matter: embed PTX instead of machine code, uncompressed, and link the
# shared CUDA runtime so that `warplens run` can stand in for it.
#
# The clone, .cutlass beside this script: one that is there is used as it is and never
# fetched again (it is only checked out at the pinned commit when it is at another). One
# that is missing is fetched: that commit alone, without history, into .cutlass.part, which
# takes the clone's place only when the fetch is whole. 5.5 s and 244 MB from github.com on
# a fast link (measured 2026-11-24); given up on after 15 minutes, or 30 s without data.
#
#   NVCC                        the compiler (default: nvcc)
#   WARPLENS_CUTLASS_URL        fetch from here instead of github.com (a mirror)
#   WARPLENS_CUTLASS_NO_FETCH   set to anything: never use the network. A missing clone,
#                               or one that lacks the commit, is then an error at once.
#                               env/remote/run.sh sets it for every stage: on a rented
#                               machine env/remote/setup.sh fetches the clone beforehand
#   WARPLENS_CUTLASS_JOBS       tutorials compiled at once (default: all four with four
#                               or more CPUs, else one). One compile peaks at 1.9 GB of
#                               memory; --programs takes 40 s with four at once on 20
#                               cores and 2 min 31 s with one (measured 2026-11-24)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Whole, not abbreviated: a server gives a single commit only when it is named in full.
# env/remote/pins.env (CUTLASS_COMMIT) names the same one; tests/test_remote_env.py checks.
COMMIT=f8acf1aa78f96ff2f1ba7662fe495e5535c461bf
URL="${WARPLENS_CUTLASS_URL:-https://github.com/NVIDIA/cutlass.git}"
NVCC="${NVCC:-nvcc}"
export GIT_TERMINAL_PROMPT=0     # a repository that asks for a password is a failed fetch

fetch_into() {  # fetch_into DIR: the pinned commit, alone, into the repository DIR
  timeout --kill-after=10 900 git -C "$1" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
      fetch --quiet --depth 1 "$URL" "$COMMIT" \
    && git -C "$1" checkout --quiet --detach FETCH_HEAD
}
no_fetch() {
  echo "error: $1, and WARPLENS_CUTLASS_NO_FETCH is set, so it is not fetched here." >&2
  echo "On a rented machine env/remote/setup.sh fetches it (run it again); a developer unsets the variable." >&2
  exit 1
}

if [[ ! -d .cutlass ]]; then
  [[ -z "${WARPLENS_CUTLASS_NO_FETCH:-}" ]] || no_fetch "there is no CUTLASS clone in $PWD/.cutlass"
  rm -rf .cutlass.part
  git init --quiet .cutlass.part
  if ! fetch_into .cutlass.part; then
    rm -rf .cutlass.part
    echo "error: could not fetch CUTLASS commit $COMMIT from $URL" >&2
    exit 1
  fi
  mv .cutlass.part .cutlass
  echo "fetched CUTLASS $COMMIT from $URL"
fi
if [[ "$(git -C .cutlass rev-parse HEAD)" != "$COMMIT" ]]; then
  # A clone with history usually has the commit; one that does not gets it added.
  if ! git -C .cutlass checkout --quiet "$COMMIT" 2>/dev/null; then
    [[ -z "${WARPLENS_CUTLASS_NO_FETCH:-}" ]] || no_fetch "the CUTLASS clone in $PWD/.cutlass does not have commit $COMMIT"
    fetch_into .cutlass || { echo "error: could not fetch CUTLASS commit $COMMIT from $URL into the clone" >&2; exit 1; }
  fi
fi

T=.cutlass/examples/cute/tutorial/blackwell
INCLUDES=(-I.cutlass/include -I.cutlass/tools/util/include -I.cutlass/examples/common -I"$T")
with_programs=0
[[ "${1:-}" == "--programs" ]] && with_programs=1
(( with_programs )) && mkdir -p bin

# build_one SOURCE NAME: NAME.ptx, and with --programs bin/NAME. Each is compiled under
# another name and renamed when it is whole, so a compile that was stopped leaves nothing
# that looks built.
build_one() {
  local source="$1" name="$2"
  "$NVCC" -arch=sm_100a -std=c++17 -O3 -DNDEBUG -w "${INCLUDES[@]}" \
    -ptx "$T/$source.cu" -o ".part.$name.ptx"
  mv -f ".part.$name.ptx" "$name.ptx"
  echo "built $name.ptx from CUTLASS ${COMMIT:0:7}"
  if (( with_programs )); then
    "$NVCC" -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared \
      -std=c++17 -O3 -DNDEBUG -w "${INCLUDES[@]}" \
      "$T/$source.cu" -o "bin/$name.part"
    mv -f "bin/$name.part" "bin/$name"
    echo "built bin/$name"
  fi
}

PAIRS=(01_mma_sm100:tutorial_01 02_mma_tma_sm100:tutorial_02
       03_mma_tma_multicast_sm100:tutorial_03 04_mma_tma_2sm_sm100:tutorial_04)
jobs="${WARPLENS_CUTLASS_JOBS:-}"
if [[ -z "$jobs" ]]; then
  jobs=1
  (( $(nproc 2>/dev/null || echo 1) >= 4 )) && jobs=${#PAIRS[@]}
fi
if (( jobs <= 1 )); then
  for pair in "${PAIRS[@]}"; do build_one "${pair%%:*}" "${pair##*:}"; done
  exit 0
fi
# At once: what each compile prints is kept and shown in order, and one failure fails the build.
logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
pids=()
for pair in "${PAIRS[@]}"; do
  ( build_one "${pair%%:*}" "${pair##*:}" ) > "$logs/${pair##*:}" 2>&1 &
  pids+=($!)
done
failed=0
for i in "${!PAIRS[@]}"; do
  wait "${pids[$i]}" || failed=1
  cat "$logs/${PAIRS[$i]##*:}"
done
if (( failed )); then
  echo "error: a CUTLASS tutorial did not compile (above)" >&2
  exit 1
fi
