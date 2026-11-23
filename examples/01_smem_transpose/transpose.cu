// Transpose a 32x32 tile through shared memory, three ways.
//
// Each CTA has 32x32 threads. Thread (x, y) writes its input element into a
// shared tile at [y][x], then reads the tile at [x][y]. The write goes along a
// row, the read goes down a column. Shared memory has 32 banks of 4 bytes, so
// what matters is which bank each lane of a warp touches:
//
//   linear    tile[32][32]   a column is 32 words in the same bank: 32-way conflict
//   padded    tile[32][33]   the extra column shifts each row by one bank: no conflict
//   swizzled  tile[32][32]   column index is XORed with the row: no conflict, no extra memory

#define TILE 32

extern "C" __global__ void transpose_linear(const float* in, float* out) {
    __shared__ float tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    const float* src = in + blockIdx.x * TILE * TILE;
    float* dst = out + blockIdx.x * TILE * TILE;
    tile[y][x] = src[y * TILE + x];
    __syncthreads();
    dst[y * TILE + x] = tile[x][y];
}

extern "C" __global__ void transpose_padded(const float* in, float* out) {
    __shared__ float tile[TILE][TILE + 1];
    int x = threadIdx.x, y = threadIdx.y;
    const float* src = in + blockIdx.x * TILE * TILE;
    float* dst = out + blockIdx.x * TILE * TILE;
    tile[y][x] = src[y * TILE + x];
    __syncthreads();
    dst[y * TILE + x] = tile[x][y];
}

extern "C" __global__ void transpose_swizzled(const float* in, float* out) {
    __shared__ float tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    const float* src = in + blockIdx.x * TILE * TILE;
    float* dst = out + blockIdx.x * TILE * TILE;
    tile[y][x ^ y] = src[y * TILE + x];
    __syncthreads();
    dst[y * TILE + x] = tile[x][y ^ x];
}
