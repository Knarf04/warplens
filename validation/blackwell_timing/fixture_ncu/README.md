# Fixtures for the ncu scripts

`run_ncu.sh` here and `validation/ampere/run_ncu.sh` need permission to read GPU performance
counters. The development machine does not give it to a normal user (`RmProfilingAdminOnly: 1`),
so the tests cannot take the profiling path against a real ncu: `tests/validation/probes/test_ncu_scripts.py`
runs the scripts against the files below instead. The path has run for real twice, as root
on the RTX 3080 Ti on 2026-11-24, with ncu 2024.1.1 and with the pinned 2026.4.0, and read
counters (`../README.md`); that was the previous version of `run_ncu.sh`, before its time
limits, groups and `STATUS.txt`. **On the development machine the version here has not run
against a real ncu that returned counters.** On the B200 (2026-11-25) it did: eleven
programs profiled (`../measured/b200/fallback-nvcc-13.3.73/ncu/`).

| File | What it is | Recorded or made up |
| --- | --- | --- |
| `permission_denied.txt` | what `ncu --csv --metrics smsp__inst_executed.sum <program>` printed on the development machine as a normal user (ncu 2024.1.1.0; the pinned 2026.4.0.0 prints the same two lines), with the program's path replaced | recorded, 2026-11-23 |
| `query_metrics_gb100.txt` | the rows of `ncu --query-metrics --chips gb100` (ncu 2026.4.0.0, the pinned release, which needs no GPU for this) for the counters the scripts ask for | recorded, 2026-11-23 |
| `../../ampere/transpose_ncu_3080ti_grid64.csv` | a complete `ncu --csv` run of `transpose_hw 64` on the RTX 3080 Ti (ncu 2024.1), from an earlier session that had permission | recorded |
| `fake_ncu.sh` | a stand-in for `ncu` that replays the files above and, for every other program, prints a counter table in ncu's CSV layout with the value `1,234` everywhere. On request it never ends for named programs (`FAKE_NCU_HANG`), ends without a table (`FAKE_NCU_FAIL`) or takes seconds per profile (`FAKE_NCU_SLEEP`), for the tests of the limits, the budget and of stopping | made up, apart from what it replays |

What the recordings established before the rental:

- Of the 19 base counter names the scripts use, the pinned ncu lists 18 for `gb100`. The
  one it does not list is `l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal`, which is not
  listed for `ga103` either (by 2024.1.1.0 or by 2026.4.0.0). On the 3080 Ti, ncu 2024.1
  accepted that name and reported `n/a` for it (the recorded CSV has those rows), so an
  unlisted name did not make the run fail there. If the B200's ncu does reject a list,
  `run_ncu.sh` retries it without the unlisted names and says so in `STATUS.txt`.
  Check again with `NCU_BIN=<ncu> ./run_ncu.sh --check-metrics gb100`.
- What ncu prints when a counter list is rejected was **not** recorded (the permission
  error comes first here); the stand-in's text for that case is invented, and the scripts
  do not parse it: they look for the `"ID"` header of the table.
