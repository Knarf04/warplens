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


def pytest_collection_modifyitems(config, items):
    """Tests marked `gpu` are deselected, not skipped, on a run that has no GPU in reach
    (CUDA_VISIBLE_DEVICES set to nothing, or no nvidia-smi): CI's full job is such a run, and
    there a skip is a failure (WARPLENS_NO_SKIPS). With a GPU they run, and skip with a reason
    for whatever else they lack."""
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and shutil.which("nvidia-smi"):
        return
    left_out = [item for item in items if item.get_closest_marker("gpu")]
    if left_out:
        config.hook.pytest_deselected(items=left_out)
        items[:] = [item for item in items if not item.get_closest_marker("gpu")]


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
