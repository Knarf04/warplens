"""The fixture and the collection rules every test gets. Builders, paths and stand-ins that
tests share are modules of tests/support, imported by name (`from tests.support import kernel`)."""

import shutil

import pytest

import warplens

# The helpers of tests/support assert too: have their failures explained like a test's.
pytest.register_assert_rewrite("tests.support")


@pytest.fixture
def sim():
    return warplens.Device("B200")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: runs on this machine's own CUDA GPU; left out where there is none")
    config.addinivalue_line("markers", "golden_replay: replays results recorded from a real GPU and committed under "
                                       "validation/, in the simulator; needs no GPU and runs everywhere")
    config.addinivalue_line("markers", "rental_archive(variable): reads a file of a rental's archive that is too "
                            "large for the repository; left out where the variable naming it is not set")


def pytest_collection_modifyitems(config, items):
    """Tests marked `gpu` are deselected, not skipped, on a run that has no GPU in reach
    (CUDA_VISIBLE_DEVICES set to nothing, or no nvidia-smi): CI's full job is such a run, and
    there a skip is a failure (WARPLENS_NO_SKIPS). With a GPU they run, and skip with a reason
    for whatever else they lack.

    Tests marked `rental_archive("VARIABLE")` are deselected the same way where VARIABLE is not
    set: what they read exists only in an archive brought back from a rented machine, and no
    install step can supply it. Where it is set they run, and skip if it names the wrong thing."""
    import os

    def left_out(item) -> bool:
        archive = item.get_closest_marker("rental_archive")
        return bool(archive and not os.environ.get(archive.args[0])) or bool(no_gpu and item.get_closest_marker("gpu"))

    no_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not shutil.which("nvidia-smi")
    deselected = [item for item in items if left_out(item)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not left_out(item)]


def pytest_sessionfinish(session, exitstatus):
    """With WARPLENS_NO_SKIPS=1 a skipped test fails the run. CI's full job sets it, so a
    missing toolchain or package there is an error instead of silently reduced coverage."""
    import os
    if os.environ.get("WARPLENS_NO_SKIPS") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        reporter.write_line(f"WARPLENS_NO_SKIPS=1: {len(skipped)} test(s) were skipped; failing the run",
                            red=True)
        for report in skipped:
            reporter.write_line(f"  {report.nodeid}: {report.longrepr[2] if isinstance(report.longrepr, tuple) else ''}")
        session.exitstatus = 1
