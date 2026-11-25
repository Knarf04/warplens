"""Where things are in the checkout the tests run from.

Every path is worked out from the location of the tests tree, never from the imported package
and never from the depth of the test that asks. CI's `dist` job replaces `<checkout>/warplens`
with a link to the installed package (.github/workflows/ci.yml): `PACKAGE` names that link, so a
test that reads a file of the package through it reads the installed one there. A test that
means the imported package, wherever it is, asks `warplens.__file__` itself.
"""

from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent
REPO = TESTS.parent
PACKAGE = REPO / "warplens"
EXAMPLES = REPO / "examples"
VALIDATION = REPO / "validation"
DOCS = REPO / "docs"
TOOLS = REPO / "tools"
ENV = REPO / "env"

# The two suites of probes for a rented B200, and what an RTX 3080 Ti recorded.
AMPERE = VALIDATION / "ampere"
BLACKWELL = VALIDATION / "blackwell"
BLACKWELL_TIMING = VALIDATION / "blackwell_timing"
