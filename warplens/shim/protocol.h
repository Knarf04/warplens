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
  WL_TENSOR_MAP = 10        /* uint32 type, rank, swizzle, fill, interleave; uint64 address; uint64 dim[5], stride[4]; uint32 box[5], step[5] -> 128 bytes */
};

/* Exported by the libcudart shim; the libcuda shim uses it too.
 * Sends one request built from `parts` and returns the reply's status. The reply payload
 * is copied into `reply` (up to `capacity` bytes) and its full length stored in `*length`.
 * A non-zero status has already been reported on stderr. */
struct wl_part { const void *data; size_t size; };
int warplens_call(uint32_t op, const struct wl_part *parts, int count,
                  void *reply, size_t capacity, size_t *length);

#endif
