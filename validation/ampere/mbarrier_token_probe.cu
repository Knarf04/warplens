// What the token of mbarrier.arrive encodes, asked of mbarrier.test_wait.
//
// One thread, one mbarrier initialised with `count` arrivals per phase, `steps` arrivals
// (at most 64). After arrival s, four words are written to out[4 * s ...]:
//   [0]  bit 63 of the token arrival s returned
//   [1]  test_wait.parity with 0, 1, 2 and 3, as bits 0 to 3
//   [2]  test_wait with the token of every arrival so far: bit j is the answer for the
//        token of arrival j (j <= s), however many phases old it is
//   [3]  test_wait with the made-up tokens 1 << b, b = 0..63: bit b is the answer
// test_wait never blocks and nothing is asynchronous, so the output is deterministic.
// The forms exist since sm_80.
extern "C" __global__ void mbarrier_token_probe(unsigned count, int steps, unsigned long long* out) {
    __shared__ unsigned long long bar;
    unsigned long long tokens[64];
    unsigned addr = (unsigned)__cvta_generic_to_shared(&bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
    for (int s = 0; s < steps; ++s) {
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(tokens[s]) : "r"(addr) : "memory");
        unsigned long long parity = 0, own = 0, made_up = 0;
        unsigned done;
        for (unsigned p = 0; p < 4; ++p) {
            asm volatile("{\n.reg .pred p;\n"
                         "mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
                         "selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(addr), "r"(p) : "memory");
            parity |= (unsigned long long)done << p;
        }
        for (int j = 0; j < 64; ++j) {
            if (j <= s) {
                asm volatile("{\n.reg .pred p;\n"
                             "mbarrier.test_wait.shared.b64 p, [%1], %2;\n"
                             "selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(addr), "l"(tokens[j]) : "memory");
                own |= (unsigned long long)done << j;
            }
            asm volatile("{\n.reg .pred p;\n"
                         "mbarrier.test_wait.shared.b64 p, [%1], %2;\n"
                         "selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(addr), "l"(1ull << j) : "memory");
            made_up |= (unsigned long long)done << j;
        }
        out[4 * s] = tokens[s] >> 63;
        out[4 * s + 1] = parity;
        out[4 * s + 2] = own;
        out[4 * s + 3] = made_up;
    }
}
