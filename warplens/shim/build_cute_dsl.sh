#!/usr/bin/env bash
# Build a runtime library for the CuTe DSL whose CUDA calls reach `warplens run`.
#
# The DSL's compiled host code calls wrappers (_cudaLibraryLoadData, _cudaLaunchKernelEx, ...)
# that the DSL ships twice: in libcute_dsl_runtime.so, linked with NVIDIA's CUDA runtime, and
# as a static archive that leaves the runtime to the linker. This links that archive, from
# the installed nvidia-cutlass-dsl package, against the libcudart shim. Nothing is copied.
#
#   warplens/shim/build_cute_dsl.sh <libcuda_dialect_runtime_static.a> <shim library directory> <output directory>
#
# The DSL takes the result through CUTE_DSL_LIBS; `warplens run` sets that.
set -euo pipefail
archive="$1"; shims="$2"; out="$3"
[[ -f "$archive" ]] || { echo "error: $archive not found" >&2; exit 1; }
runtime="$(ls "$shims"/libcudart.so.* | head -1)"
cc="${CC:-cc}"
mkdir -p "$out"
stage="$(mktemp -d "$out/.build-XXXXXX")"
trap 'rm -rf "$stage"' EXIT
# Runtime functions the wrappers name and the shim does not implement fail when called,
# instead of keeping the library from loading.
comm -23 <(nm "$archive" | awk '$1 == "U" && $2 ~ /^cuda[A-Z]/ {print $2}' | sort -u) \
         <(nm -D --defined-only "$runtime" | awk '{print $3}' | sed 's/@.*//' | sort -u) > "$stage/missing"
{
  echo '#include <stdio.h>'
  while read -r name; do
    printf 'int %s(void) { fprintf(stderr, "warplens: the CUDA runtime function %s is not implemented\\n"); return 801; }\n' "$name" "$name"
  done < "$stage/missing"
} > "$stage/missing.c"   # 801 is cudaErrorNotSupported
"$cc" -shared -fPIC -O2 -o "$stage/libcute_dsl_runtime.so" -Wl,-soname,libcute_dsl_runtime.so \
  -Wl,--whole-archive "$archive" -Wl,--no-whole-archive "$stage/missing.c" \
  -L"$shims" -l:"$(basename "$runtime")" -Wl,-rpath,"$(cd "$shims" && pwd)"
mv -f "$stage/libcute_dsl_runtime.so" "$out/"
echo "built $out/libcute_dsl_runtime.so ($(wc -l < "$stage/missing") runtime function(s) not implemented)"
