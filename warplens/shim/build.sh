#!/usr/bin/env bash
# Build the shim libraries that `warplens run` puts in front of the real CUDA runtime.
# Needs a C compiler and the CUDA toolkit headers (they are included, not copied).
#
#   warplens/shim/build.sh [include directory of the CUDA toolkit]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
include="${1:-}"
if [[ -z "$include" ]]; then
  nvcc_path="$(command -v nvcc || true)"
  [[ -n "$nvcc_path" ]] || { echo "error: nvcc not found; pass the CUDA include directory" >&2; exit 1; }
  root="$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")"
  for candidate in "$root/targets/x86_64-linux/include" "$root/include"; do
    [[ -f "$candidate/cuda_runtime_api.h" ]] && include="$candidate" && break
  done
fi
[[ -f "$include/cuda_runtime_api.h" ]] || { echo "error: cuda_runtime_api.h not found in '$include'" >&2; exit 1; }
major="$(sed -n 's/^#define CUDART_VERSION *\([0-9]*\).*/\1/p' "$include/cuda_runtime_api.h" | head -1)"
major="$((major / 1000))"
cc="${CC:-cc}"
out="$here/lib"
mkdir -p "$out"
# Programs ask for symbols of version "libcudart.so.N", so define that version node.
printf 'libcudart.so.%s { global: *; };\n' "$major" > "$out/cudart.map"
"$cc" -shared -fPIC -O2 -Wall -Wextra -I"$include" -I"$here" "$here/cudart_shim.c" \
  -Wl,-soname,"libcudart.so.$major" -Wl,--version-script,"$out/cudart.map" -lpthread -ldl \
  -o "$out/libcudart.so.$major"
"$cc" -shared -fPIC -O2 -Wall -Wextra -I"$include" -I"$here" "$here/cuda_shim.c" \
  -Wl,-soname,libcuda.so.1 -L"$out" -l:"libcudart.so.$major" -Wl,-rpath,'$ORIGIN' \
  -o "$out/libcuda.so.1"
# No libcudart.so / libcuda.so links here: programs must link against the real libraries.
rm -f "$out/libcudart.so" "$out/libcuda.so"
echo "built $out/libcudart.so.$major and $out/libcuda.so.1 (CUDA $major headers from $include)"
