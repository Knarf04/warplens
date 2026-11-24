// Host-only capture hooks for the pinned CUTLASS tutorials. Generated copies include this;
// the upstream checkout and the device kernel bodies are never changed.
#pragma once
#include <cuda.h>
#include <cutlass/cluster_launch.hpp>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace wl_capture {
inline const char* directory() { return std::getenv("WARPLENS_CAPTURE_RAW_DIR"); }
inline std::string path(const char* name) { return std::string(directory()) + "/" + name; }
inline std::ofstream file(const char* name, bool append = false) {
  std::ofstream out(path(name), std::ios::binary | (append ? std::ios::app : std::ios::trunc));
  if (!out) throw std::runtime_error("cannot open capture file " + path(name));
  out.exceptions(std::ios::badbit | std::ios::failbit);
  return out;
}
inline std::string hex(const void* ptr, size_t bytes) {
  auto p = static_cast<const unsigned char*>(ptr);
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes; ++i) out << std::setw(2) << unsigned(p[i]);
  return out.str();
}
inline void binary(const char* name, const void* data, size_t bytes) {
  auto out = file(name);
  out.write(static_cast<const char*>(data), bytes);
}
template<class T> void array(std::ostream& out, const T* data, size_t n) {
  out << '[';
  for (size_t i = 0; i < n; ++i) out << (i ? "," : "") << data[i];
  out << ']';
}
inline void dims(std::ostream& out, dim3 v) { unsigned d[] = {v.x, v.y, v.z}; array(out, d, 3); }

template<class A, class B, class C, class D>
void buffers(int m, int n, int k, const A* a, const B* b, const C* c, const D* d,
             const void* da, const void* db, const void* dc, const void* dd) {
  if (!directory()) return;
  static_assert(sizeof(A) == 2 && sizeof(B) == 2 && sizeof(C) == 4 && sizeof(D) == 4);
  binary("a.input.bin", a, size_t(m) * k * sizeof(A));
  binary("b.input.bin", b, size_t(n) * k * sizeof(B));
  binary("c.input.bin", c, size_t(m) * n * sizeof(C));
  binary("d.input.bin", d, size_t(m) * n * sizeof(D));
  auto out = file("buffers.json");
  out << "{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
      << ",\"pointers\":{\"a\":" << reinterpret_cast<uintptr_t>(da)
      << ",\"b\":" << reinterpret_cast<uintptr_t>(db)
      << ",\"c\":" << reinterpret_cast<uintptr_t>(dc)
      << ",\"d\":" << reinterpret_cast<uintptr_t>(dd) << "}}\n";
}
template<class D> void result(const D* output, const D* reference, size_t count, bool success) {
  if (!directory()) return;
  binary("d.output.bin", output, count * sizeof(D));
  binary("d.reference.bin", reference, count * sizeof(D));
  auto out = file("result.json");
  out << "{\"reference_passed\":" << (success ? "true" : "false") << "}\n";
}

template<class... Args>
cutlass::Status launch(const cutlass::ClusterLaunchParams& params, const void* kernel, Args&&... args) {
  if (directory()) {
    auto out = file("launch.json");
    out << "{\"grid\":"; dims(out, params.grid_dims);
    out << ",\"block\":"; dims(out, params.block_dims);
    out << ",\"cluster\":"; dims(out, params.cluster_dims);
    out << ",\"smem\":" << params.smem_size_in_bytes << ",\"args\":[";
    size_t i = 0;
    ((out << (i++ ? "," : "") << "{\"hex\":\"" << hex(&args, sizeof(args)) << "\"}"), ...);
    out << "]}\n";
  }
  return cutlass::launch_kernel_on_cluster(params, kernel, std::forward<Args>(args)...);
}

template<class Atom> void binding(unsigned argument, const Atom& atom) {
  if (!directory()) return;
  auto desc = atom.get_tma_descriptor();
  auto offset = reinterpret_cast<const char*>(desc) - reinterpret_cast<const char*>(&atom);
  if (offset < 0 || size_t(offset) + 128 > sizeof(atom))
    throw std::runtime_error("tensor map is not embedded in the launch argument");
  auto out = file("bindings.jsonl", true);
  out << "{\"argument\":" << argument << ",\"offset\":" << offset
      << ",\"hex\":\"" << hex(desc, 128) << "\"}\n";
}

inline void tensor_map(const void* desc, int data_type, unsigned rank, const void* address,
                       const uint64_t* shape, const uint64_t* strides, const uint32_t* box,
                       const uint32_t* steps, int interleave, int swizzle, int l2, int fill,
                       int status) {
  if (!directory()) return;
  if (status != CUDA_SUCCESS) throw std::runtime_error("tensor-map encoding failed during capture");
  auto out = file("tensor_maps.jsonl", true);
  out << "{\"hex\":\"" << hex(desc, 128) << "\",\"address\":" << reinterpret_cast<uintptr_t>(address)
      << ",\"data_type\":" << data_type << ",\"rank\":" << rank << ",\"dims\":";
  array(out, shape, rank); out << ",\"strides\":"; array(out, strides, rank - 1);
  out << ",\"box\":"; array(out, box, rank); out << ",\"steps\":"; array(out, steps, rank);
  out << ",\"interleave\":" << interleave << ",\"swizzle\":" << swizzle
      << ",\"l2_promotion\":" << l2 << ",\"nan_fill\":" << (fill ? "true" : "false") << "}\n";
}
}  // namespace wl_capture
