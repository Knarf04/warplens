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
/* The shared memory per CTA that needs no opt-in (what the device reports as its shared
 * memory per block): DEFAULT_SMEM_BYTES of warplens/units/smem.py, which also holds the rule
 * for what a launch may use (`default_max_dynamic`). The server applies it. */
#define WL_DEFAULT_SMEM_BYTES (48 * 1024)

/* The two function attributes that can be set, which have the same numbers in the runtime's
 * and the driver's enumeration. */
#define WL_ATTRIBUTE_MAX_DYNAMIC_SMEM 8  /* cudaFuncAttributeMaxDynamicSharedMemorySize */
#define WL_ATTRIBUTE_SMEM_CARVEOUT 9     /* cudaFuncAttributePreferredSharedMemoryCarveout */
/* cudaFuncAttributeNonPortableClusterSizeAllowed, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED.
 * A B200 refuses a cluster of more than 8 CTAs (cudaErrorInvalidClusterSize) unless the
 * program set this to 1 for the kernel first; with it, 16 launch and 32 are refused
 * (validation/blackwell/measured/b200/cluster/max_size.txt and cluster-optin/). The server
 * keeps it per kernel (WL_SET_ATTRIBUTE) and holds launches to it. */
#define WL_ATTRIBUTE_NONPORTABLE_CLUSTER 14

/* Whether a function attribute may be set to `value` whatever the kernel is, as
 * cudaFuncSetAttribute and cuFuncSetAttribute decide it: the maximum dynamic shared memory is
 * 0 to WL_MAX_SMEM_BYTES and the preferred carveout -1 (default) to 100 percent. An RTX 3080
 * Ti refuses other values of the two, the attributes that can only be read (0 to 7) and
 * numbers that name no attribute with cudaErrorInvalidValue / CUDA_ERROR_INVALID_VALUE (at
 * its own opt-in maximum in place of ours). A maximum that passes here is then held against
 * the kernel's static shared memory by the server (WL_SET_ATTRIBUTE). Of the cluster
 * attributes (10 to 17), the non-portable cluster size (14) goes to the server too, any
 * non-zero value allowing it; the others (required cluster shape, scheduling policy, shared
 * memory mode) are accepted unchecked, as before: the CuTe DSL sets them, and what a GPU
 * with clusters says to each value was not measured. */
static inline int wl_function_attribute_is_valid(int attribute, int value) {
  if (attribute == WL_ATTRIBUTE_MAX_DYNAMIC_SMEM) return value >= 0 && value <= WL_MAX_SMEM_BYTES;
  if (attribute == WL_ATTRIBUTE_SMEM_CARVEOUT) return value >= -1 && value <= 100;
  return attribute >= 10 && attribute <= 17;
}

/* Statuses of a WL_LAUNCH reply (the runtime's error numbers). The first three mean the
 * launch was refused before the kernel ran; anything else is a kernel that failed. */
#define WL_INVALID_VALUE 1           /* cudaErrorInvalidValue: shared memory over the limit, or over what
                                        the kernel was opted in to */
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
  WL_REGISTER_MODULE = 11,  /* PTX text, a fat binary, or the first bytes of machine code -> uint32 module */
  WL_SET_ATTRIBUTE = 12,    /* uint32 function, int32 attribute, int32 value (the maximum dynamic shared memory, or non-portable cluster sizes) */
  WL_GET_ATTRIBUTES = 13    /* uint32 function                      -> uint32 static shared bytes, uint32 maximum dynamic */
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
/* Sets a function attribute of the kernel named by `key`, as cudaFuncSetAttribute does, and
 * returns the runtime's error number: 0, cudaErrorInvalidDeviceFunction for a key that names
 * no kernel, or cudaErrorInvalidValue. The maximum dynamic shared memory is what a later
 * launch of that kernel is held to. */
int warplens_set_function_attribute(const void *key, int attribute, int value);
/* The static shared memory of the kernel named by `key` and the most dynamic shared memory a
 * launch of it may use now. Returns 0, or cudaErrorInvalidDeviceFunction. */
int warplens_function_shared_memory(const void *key, int *fixed, int *max_dynamic);
/* The value of a device attribute (the runtime's and the driver's enumerations agree), or 0
 * for one the simulated device does not describe. */
int warplens_device_attribute(int attribute);

#endif
