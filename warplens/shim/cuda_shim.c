/* A stand-in for libcuda (the driver API): only the calls programs make directly. */
#include <string.h>

#include <cuda.h>

#include "protocol.h"

CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
                                cuuint32_t tensorRank, void *globalAddress,
                                const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
                                const cuuint32_t *boxDim, const cuuint32_t *elementStrides,
                                CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                                CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
  (void)l2Promotion;
  if (tensorRank < 1 || tensorRank > 5) return CUDA_ERROR_INVALID_VALUE;
  uint32_t head[5] = {(uint32_t)tensorDataType, tensorRank, (uint32_t)swizzle, (uint32_t)oobFill,
                      (uint32_t)interleave};
  uint64_t address = (uint64_t)globalAddress, dims[5] = {0}, strides[4] = {0};
  uint32_t box[5] = {0}, steps[5] = {0};
  for (cuuint32_t i = 0; i < tensorRank; ++i) {
    dims[i] = globalDim[i]; box[i] = boxDim[i]; steps[i] = elementStrides[i];
    if (i + 1 < tensorRank) strides[i] = globalStrides[i];
  }
  struct wl_part parts[6] = {{head, sizeof head}, {&address, sizeof address}, {dims, sizeof dims},
                             {strides, sizeof strides}, {box, sizeof box}, {steps, sizeof steps}};
  memset(tensorMap, 0, sizeof *tensorMap);
  return warplens_call(WL_TENSOR_MAP, parts, 6, tensorMap, sizeof *tensorMap, NULL)
             ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult cuInit(unsigned int flags) { (void)flags; return CUDA_SUCCESS; }
