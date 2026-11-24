/* The wire protocol between the shim libraries and `warplens run`.
 *
 * Request:  uint32 op, uint32 length, payload.
 * Reply:    int32 status (0 = ok), uint32 length, payload (an error message if status != 0).
 * All integers are little-endian. One request is answered before the next is sent.
 */
#ifndef WARPLENS_PROTOCOL_H
#define WARPLENS_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

/* Largest shared memory per CTA, which a program may be told and a launch may use. The
 * simulator enforces MAX_SMEM_BYTES of warplens/units/smem.py; the two must be equal. */
#define WL_MAX_SMEM_BYTES (227 * 1024)

/* Whether a function attribute may be set to `value`, as cudaFuncSetAttribute and
 * cuFuncSetAttribute decide it (the two enumerations agree on these numbers): the maximum
 * dynamic shared memory is 0 to WL_MAX_SMEM_BYTES and the preferred carveout -1 (default) to
 * 100 percent. An RTX 3080 Ti refuses other values of the two, the attributes that can only be
 * read (0 to 7) and numbers that name no attribute with cudaErrorInvalidValue /
 * CUDA_ERROR_INVALID_VALUE (at its own opt-in maximum in place of ours). The cluster
 * attributes (10 to 17: required cluster shape, non-portable sizes, scheduling policy, shared
 * memory mode) are accepted unchecked, as before: the CuTe DSL sets them, and what a GPU
 * with clusters says to each value was not measured. */
static inline int wl_function_attribute_is_valid(int attribute, int value) {
  if (attribute == 8) return value >= 0 && value <= WL_MAX_SMEM_BYTES; /* ...MaxDynamicSharedMemorySize */
  if (attribute == 9) return value >= -1 && value <= 100;              /* ...PreferredSharedMemoryCarveout */
  return attribute >= 10 && attribute <= 17;
}

/* Statuses of a WL_LAUNCH reply (the runtime's error numbers). The first three mean the
 * launch was refused before the kernel ran; anything else is a kernel that failed. */
#define WL_INVALID_VALUE 1           /* cudaErrorInvalidValue: shared memory over the limit */
#define WL_INVALID_CONFIGURATION 9   /* cudaErrorInvalidConfiguration: block or grid dimensions */
#define WL_INVALID_CLUSTER_SIZE 912  /* cudaErrorInvalidClusterSize */

enum {
  WL_REGISTER_FATBIN = 1,   /* fat binary bytes                      -> uint32 module            */
  WL_REGISTER_FUNCTION = 2, /* uint32 module, name                   -> uint32 function, uint32 n, uint32 size[n] */
  WL_MALLOC = 3,            /* uint64 bytes                          -> uint64 address           */
  WL_FREE = 4,              /* uint64 address                                                    */
  WL_WRITE = 5,             /* uint64 address, bytes                                             */
  WL_READ = 6,              /* uint64 address, uint64 bytes          -> bytes                    */
  WL_COPY = 7,              /* uint64 dst, uint64 src, uint64 bytes                              */
  WL_FILL = 8,              /* uint64 address, uint64 bytes, uint32 value                        */
  WL_LAUNCH = 9,            /* uint32 function, grid[3], block[3], cluster[3], uint64 smem, argument bytes -> printed text */
  WL_TENSOR_MAP = 10,       /* uint32 type, rank, swizzle, fill, interleave; uint64 address; uint64 dim[5], stride[4]; uint32 box[5], step[5] -> 128 bytes */
  WL_REGISTER_MODULE = 11   /* PTX text, a fat binary, or the first bytes of machine code -> uint32 module */
};

/* Exported by the libcudart shim; the libcuda shim uses it too.
 * Sends one request built from `parts` and returns the reply's status. The reply payload
 * is copied into `reply` (up to `capacity` bytes) and its full length stored in `*length`.
 * A non-zero status has already been reported on stderr. */
struct wl_part { const void *data; size_t size; };
int warplens_call(uint32_t op, const struct wl_part *parts, int count,
                  void *reply, size_t capacity, size_t *length);

/* Also exported by the libcudart shim, for modules a program loads itself (the driver API,
 * and the runtime's cudaLibrary calls) instead of through code nvcc generated.
 * warplens_load_module takes PTX text, a fat binary or machine code and returns a handle, or
 * NULL after reporting why not. Machine code is accepted only when `warplens run` knows where
 * the compiler kept the PTX; kernels are then found there by name.
 * warplens_module_kernel returns the address that names the kernel in a launch, or NULL. */
void *warplens_load_module(const void *image);
void *warplens_module_kernel(void *module, const char *name);
/* Launches the kernel named by `key` and prints what it printed. Returns the reply's status.
 * A zero cluster means no cluster. */
int warplens_launch(const void *key, const uint32_t grid[3], const uint32_t block[3],
                    const uint32_t cluster[3], uint64_t shared_memory, void **arguments);
/* The value of a device attribute (the runtime's and the driver's enumerations agree), or 0
 * for one the simulated device does not describe. */
int warplens_device_attribute(int attribute);

#endif
