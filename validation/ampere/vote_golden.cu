// Host program: run vote_probe on the GPU and print one line per launch:
//
//   mode threads roles predicates mask[0..threads) out[0..3 threads)
//
// `roles` and `predicates` are one digit per thread; masks and results are 32-bit hex. A
// result of eeeeeeee is a word the thread did not write. See vote_probe.cu for the roles,
// the modes and the three words per thread.
//
//   vote_golden              the golden file: the launches below, each for every mode
//   vote_golden other        how many launches the second list has
//   vote_golden other K      launch K of the second list; the line is printed up to the masks
//                            before the launch, so that a launch that does not return shows
//                            which one it was (run_vote_other.sh runs them under a timeout)
//
// The golden launches:
//
//   one full warp, every lane in the mask         predicates all 0, all 1, one set, one clear,
//                                                 alternating, pseudo-random
//   partial warps (1, 5, 31, 33, 48, 64 threads)  the mask names all 32 lanes, or the lanes
//                                                 that exist
//   lanes that returned before the vote           the mask names them too, or only the voters
//   lanes on a branch without a vote              each voter's mask is the voters
//   lanes on a branch with a ballot of their own  each group's mask is its own lanes
//   a mixture of the four roles                   likewise
//   several groups at the same instruction        halves, even and odd lanes, four groups of
//                                                 eight, every lane alone: each lane's mask is
//                                                 its group
//
// The second list is what PTX does not allow or what does not return on an RTX 3080 Ti; its
// outcomes are recorded in vote_other_3080ti.txt and are not golden results:
//
//   the mask names a lane that is on another branch (roles 2 and 3, the mask 0xffffffff)
//   the mask leaves out the caller: the low half only, no lane, the next lane only, every
//   lane but the caller
//
// With -DVOTE_LEGACY (built for sm_60) the modes are 5 to 9, the forms without .sync.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "vote_probe.cu"

struct VoteCase {
    int n;
    std::vector<int> role, pred;
    std::vector<unsigned> mask;
};

static unsigned vote_state;
static unsigned vote_next() { vote_state = vote_state * 1664525u + 1013904223u; return vote_state >> 8; }

// The lanes of thread t's warp that have its role.
static unsigned vote_group(const std::vector<int>& role, int t) {
    unsigned bits = 0;
    for (int u = t / 32 * 32; u < (int)role.size() && u < t / 32 * 32 + 32; ++u)
        if (role[u] == role[t]) bits |= 1u << (u % 32);
    return bits;
}

enum { VOTE_ALL_LANES, VOTE_OWN_GROUP };

// predicates: 0 all clear, 1 all set, 2 set for role 0, 3 set for the other roles, 4 pseudo-random
static void vote_add(std::vector<VoteCase>& cases, const std::vector<int>& role, int masks, int preds) {
    VoteCase c;
    c.n = (int)role.size();
    c.role = role;
    for (int t = 0; t < c.n; ++t) {
        c.mask.push_back(masks == VOTE_ALL_LANES ? 0xffffffffu : vote_group(role, t));
        c.pred.push_back(preds == 0 ? 0 : preds == 1 ? 1 : preds == 2 ? role[t] == 0 :
                         preds == 3 ? role[t] != 0 : vote_next() & 1);
    }
    cases.push_back(c);
}

static std::vector<VoteCase> vote_cases() {
    std::vector<VoteCase> cases;
    vote_state = 2026u;
    auto every = [&](const std::vector<int>& role, int masks) {
        for (int preds : {0, 1, 2, 3, 4, 4}) vote_add(cases, role, masks, preds);
    };

    // one full warp
    const std::vector<int> voters(32, 0);
    vote_add(cases, voters, VOTE_ALL_LANES, 0);
    vote_add(cases, voters, VOTE_ALL_LANES, 1);
    for (int lane : {0, 13, 31})
        for (int value : {0, 1}) {
            vote_add(cases, voters, VOTE_ALL_LANES, 1 - value);
            cases.back().pred[lane] = value;
        }
    for (int odd : {0, 1}) {
        vote_add(cases, voters, VOTE_ALL_LANES, 0);
        for (int t = 0; t < 32; ++t) cases.back().pred[t] = (t & 1) == odd;
    }
    for (int k = 0; k < 6; ++k) vote_add(cases, voters, VOTE_ALL_LANES, 4);

    // partial warps
    for (int n : {1, 5, 31, 33, 48, 64})
        for (int masks : {VOTE_ALL_LANES, VOTE_OWN_GROUP})
            for (int preds : {0, 1, 4, 4}) vote_add(cases, std::vector<int>(n, 0), masks, preds);

    // some lanes return (1), branch away without a vote (2), or take their own ballot (3)
    for (int other : {1, 2, 3})
        for (int n : {32, 48}) {
            std::vector<int> role(n, 0);
            auto with = [&](int from, int to) {
                std::vector<int> r = role;
                for (int t = from; t < to && t < n; ++t) r[t] = other;
                return r;
            };
            std::vector<int> all_but_one(n, other), scattered(n, 0);
            all_but_one[7] = 0;
            for (int t = 0; t < n; ++t) if (vote_next() & 1) scattered[t] = other;
            for (const std::vector<int>& r : {with(0, 1), with(0, 16), with(16, 32), with(31, 40),
                                              all_but_one, scattered}) {
                if (other == 1) every(r, VOTE_ALL_LANES);  // a mask may name a lane that returned
                every(r, VOTE_OWN_GROUP);
            }
        }

    // a mixture
    for (int n : {32, 48, 64})
        for (int k = 0; k < 4; ++k) {
            std::vector<int> role(n);
            for (int t = 0; t < n; ++t) role[t] = vote_next() % 4;
            every(role, VOTE_OWN_GROUP);
        }

    // several groups at the same instruction, each lane with its group's mask
    for (int grouping = 0; grouping < 4; ++grouping)
        for (int preds = 0; preds < 5; ++preds) {
            vote_add(cases, voters, VOTE_ALL_LANES, preds < 2 ? preds : 4);
            VoteCase& c = cases.back();
            for (int t = 0; t < 32; ++t) {
                c.mask[t] = grouping == 0 ? (t < 16 ? 0x0000ffffu : 0xffff0000u) :
                            grouping == 1 ? (t % 2 ? 0xaaaaaaaau : 0x55555555u) :
                            grouping == 2 ? 0xffu << (t / 8 * 8) : 1u << t;
                if (preds == 2) c.pred[t] = (c.mask[t] & 1) != 0;  // the group of lane 0 only
            }
        }
    return cases;
}

// Not golden: see the top of the file.
static std::vector<VoteCase> vote_other_cases() {
    std::vector<VoteCase> cases;
    vote_state = 86u;
    const std::vector<int> voters(32, 0);
    // the mask names lanes on another branch
    for (int other : {2, 3})
        for (int half = 0; half < 2; ++half) {
            std::vector<int> role(32, 0);
            for (int t = 0; t < (half ? 16 : 1); ++t) role[t] = other;
            for (int preds : {1, 4}) vote_add(cases, role, VOTE_ALL_LANES, preds);
        }
    // the mask leaves out the caller
    for (int kind = 0; kind < 4; ++kind)
        for (int preds : {0, 1, 4}) {
            vote_add(cases, voters, VOTE_ALL_LANES, preds);
            VoteCase& c = cases.back();
            for (int t = 0; t < 32; ++t)
                c.mask[t] = kind == 0 ? 0x0000ffffu : kind == 1 ? 0u :
                            kind == 2 ? 1u << ((t + 1) % 32) : ~(1u << t);
        }
    return cases;
}

int main(int argc, char** argv) {
#ifdef VOTE_LEGACY
    const int first_mode = 5;
#else
    const int first_mode = 0;
#endif
    const bool other = argc > 1 && !strcmp(argv[1], "other");
    std::vector<VoteCase> cases = other ? vote_other_cases() : vote_cases();
    const int launches = 5 * (int)cases.size();
    if (other && argc < 3) {
        printf("%d\n", launches);
        return 0;
    }
    const int only = other ? atoi(argv[2]) : -1;
    const int top = 64;
    unsigned *d_out, *d_mask;
    int *d_pred, *d_role;
    CHECK(cudaMalloc(&d_out, 3 * top * 4)); CHECK(cudaMalloc(&d_mask, top * 4));
    CHECK(cudaMalloc(&d_pred, top * 4)); CHECK(cudaMalloc(&d_role, top * 4));
    for (int launch = 0; launch < launches; ++launch) {
        if (other && launch != only) continue;
        const int mode = first_mode + launch / (int)cases.size();
        const VoteCase& c = cases[launch % cases.size()];
        std::vector<unsigned> out(3 * c.n, 0xeeeeeeeeu);
        std::string roles, preds;
        for (int t = 0; t < c.n; ++t) { roles += char('0' + c.role[t]); preds += char('0' + c.pred[t]); }
        printf("%d %d %s %s", mode, c.n, roles.c_str(), preds.c_str());
        for (unsigned m : c.mask) printf(" %08x", m);
        if (other) fflush(stdout);
        CHECK(cudaMemcpy(d_out, out.data(), 3 * c.n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_mask, c.mask.data(), c.n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_pred, c.pred.data(), c.n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_role, c.role.data(), c.n * 4, cudaMemcpyHostToDevice));
        vote_probe<<<1, c.n>>>(d_out, d_pred, d_mask, d_role, mode);
        CHECK_LAUNCH();
        CHECK(cudaMemcpy(out.data(), d_out, 3 * c.n * 4, cudaMemcpyDeviceToHost));
        for (unsigned v : out) printf(" %08x", v);
        printf("\n");
    }
    return 0;
}
