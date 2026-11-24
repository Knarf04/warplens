// Atomic updates: validation/ampere/atom_golden.cu, rebuilt for this GPU and rerun.
//
// The kernel (validation/ampere/atom_probe.cu; included, not copied) has 4 CTAs of 128
// threads update the same global words, and each CTA's shared words, with every atomic
// function of CUDA C++: add, sub, exchange, min, max, and, or, xor, compare-and-swap,
// increment and decrement on 32- and 64-bit integers, add and exchange on float, add on
// double, the _block and _system variants, and pointers that are shared for some lanes of a
// warp and global for others. The host program prints only what is the same for every order
// in which the threads run (sums, extrema, masks, sorted tickets, an exchanged word folded
// into the sum of what the exchanges returned), so the output is deterministic on a GPU.
//
// The expected output is validation/ampere/atom_golden.txt, what an RTX 3080 Ti printed, and
// the committed prediction is that file, byte for byte (tests/test_ampere_replay.py).
//
// What differs between the builds is the PTX the compiler makes of this source, not the
// source: each toolkit spells `atom` its own way for its target (scopes, orderings). The
// simulator's prediction is of what nvcc 13.5 emits for sm_100a; the Ampere build
// (build.sh --ampere: nvcc 12.4, sm_86) shows the program still prints the golden file there.
//
//   ampere_atom
#include "../common.h"

#define CHECK(call) HOST_CHECK(call)
#define CHECK_LAUNCH() probe_sync("atom_probe")

#include "../../ampere/atom_golden.cu"
