"""Tests of the evidence: what was recorded from hardware, and the harness that records more.

Two kinds of test here have to do with a GPU, and they are kept apart:

    golden_replay/   OFFLINE. Replays results that a real GPU produced and that are committed
                     under validation/ (the RTX 3080 Ti's golden files, the B200's measured
                     outputs and captures) in the simulator. No GPU is involved: these run
                     everywhere, CI's GPU-less jobs included. Marked `golden_replay`.
    live_gpu/        LIVE. Runs the capture tooling on this machine's own CUDA GPU. Marked `gpu`:
                     deselected where no GPU is in reach (tests/conftest.py), and skipped with
                     a reason for whatever else they lack.

The rest needs no GPU either and replays nothing:

    capture/         capture bundles and recipes: completeness, integrity, replay without the
                     compiler that made them
    probes/          the two probe suites (validation/blackwell, validation/blackwell_timing):
                     their committed predictions, their runners, comparison and health checks
    remote/          the scripts for the rented machine (env/remote), on throwaway checkouts
                     with stand-in tools

Some comparisons with recorded hardware results are kept beside the behavior they pin down and
not here: the TF32 conversions (tests/numerics/test_tf32.py), the MMA sums of a B200
(tests/memory/test_tcgen05.py), `ldmatrix` in a cluster (tests/memory/test_matrix_layout_regressions.py).
They are offline too.
"""
