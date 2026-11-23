// cp.async with a source size: lane l copies 16 bytes from global to shared memory but
// reads only src_size[l] of them. out[l][0..15] are the 16 shared bytes afterwards.
// The shared buffer is filled with 0xEE first, to tell "zeroed" from "left alone".
extern "C" __global__ void cpasync_probe(const unsigned char* in, const unsigned* src_size,
                                         unsigned char* out) {
    __shared__ alignas(16) unsigned char buf[32][16];
    int lane = threadIdx.x;
    for (int k = 0; k < 16; ++k) buf[lane][k] = 0xEE;
    __syncthreads();
    unsigned dst = (unsigned)__cvta_generic_to_shared(&buf[lane][0]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(in + lane * 16),
                 "r"(src_size[lane]) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int k = 0; k < 16; ++k) out[lane * 16 + k] = buf[lane][k];
}
