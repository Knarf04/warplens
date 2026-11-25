# Releasing warplens

How a version is cut, and what is checked before it is. A release is a `v<version>` tag and
the sdist and wheel built from that tag, uploaded to PyPI. Uploading is done by hand: no CI
job publishes anything.

Cutting a release, and `1.0.0` in particular, is the owner's decision. Nothing here is done
by a routine commit, and green checks are a condition for a release, not a reason for one.

## The version

The version is `__version__` in `warplens/__init__.py` and nowhere else. `pyproject.toml`
reads it from there when a wheel or an sdist is built, so the bump is one line:

```python
__version__ = "1.0.0"
```

It is written in the form PyPI shows (PEP 440): `1.0.0`, a release candidate `1.0.0rc1`, a
development version `1.0.0.dev0`. `PLAN.md` names two versions and no others: `0.0.1`, a
placeholder to reserve the PyPI name, and `1.0`, the public release. The tree is at `0.0.1`.

## Before the release

1. What `PLAN.md` still lists under "Still owed for v1.0" is done, or the owner has decided to
   release without it. What has and has not been checked against hardware is said in the
   README ("Status") and in section 2 of `docs/guide.md`; the release does not change that.
2. CI is green on the commit to be tagged, in all four jobs: `test` (the core suite on each
   supported Python), `dist` (the wheel and the sdist, installed in place of the checkout),
   `triton`, and `full` (everything, with no test allowed to skip).
3. `README.md` reads correctly as the PyPI project page. It is the long description, and PyPI
   does not resolve links that are relative to the repository, so the README links to the
   repository's files by full URL (`https://github.com/Knarf04/warplens/blob/main/...`);
   `tests/documentation/test_guide.py` fails on a relative one and checks that each file is there.
4. The first upload of a name to PyPI is permanent, and a version cannot be uploaded twice:
   a mistake in an uploaded file is fixed by a new version, not by a new file.

## The steps

Run these from a fresh clone of the commit that CI passed, not from a working tree. A working
tree holds build outputs, and a `warplens.egg-info` left by an earlier build makes setuptools
put the files it lists into the sdist again, whatever `MANIFEST.in` says.

1. Bump `__version__` in `warplens/__init__.py` (the one line above).
2. In `CHANGELOG.md`, give the version its section: `## <version> (<date>)`, with what changed
   since the last release. `tests/packaging/test_packaging.py` fails while the section is missing.
3. For `1.0.0`: in `pyproject.toml`, change the classifier `Development Status :: 3 - Alpha`
   to the status the owner chooses (`5 - Production/Stable` or `4 - Beta`), and strike
   "Reserve the PyPI name" in `PLAN.md` once the upload is done.
4. Commit, open a pull request, wait for CI, merge.
5. Build, in a fresh clone of the merged commit and a virtual environment of its own:

   ```bash
   git clone https://github.com/Knarf04/warplens warplens-release && cd warplens-release
   python -m venv /tmp/release-build && /tmp/release-build/bin/pip install build twine
   /tmp/release-build/bin/python -m build            # dist/warplens-<version>.tar.gz and -py3-none-any.whl
   /tmp/release-build/bin/twine check --strict dist/*
   ```

6. Check the files that were built, the very ones that will be uploaded. Install the wheel
   into a clean environment, outside the clone, and use it:

   ```bash
   python -m venv /tmp/release-check && /tmp/release-check/bin/pip install dist/warplens-*.whl
   cd /tmp
   /tmp/release-check/bin/warplens --help
   /tmp/release-check/bin/python -c "import warplens; print(warplens.__version__, warplens.__file__)"
   /tmp/release-check/bin/python ~/warplens-release/examples/00_vecadd/run.py     # matches NumPy: True
   /tmp/release-check/bin/python ~/warplens-release/examples/02_tma_copy/run.py   # output correct
   ```

   The version printed is the one being released, and the file is under
   `/tmp/release-check`. On a machine with `nvcc` and a C compiler, also run a compiled
   program, which makes the installed package compile its shim libraries from its own
   sources (`XDG_CACHE_HOME` set to an empty directory forces the build):

   ```bash
   nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog
   XDG_CACHE_HOME=$(mktemp -d) /tmp/release-check/bin/warplens run ./prog
   ```

   The same with the sdist in a second environment (`pip install dist/warplens-*.tar.gz`).
   CI's `dist` job does all of this except the compiled program, on every push; this step is
   about the exact files of this release.
7. Tag the commit and push the tag:

   ```bash
   git tag -a v<version> -m "warplens <version>" && git push origin v<version>
   ```

8. Upload, first to TestPyPI, then to PyPI, and install from each to see that it works:

   ```bash
   /tmp/release-build/bin/twine upload --repository testpypi dist/*
   python -m venv /tmp/from-testpypi && /tmp/from-testpypi/bin/pip install \
       --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ warplens==<version>
   /tmp/release-build/bin/twine upload dist/*
   python -m venv /tmp/from-pypi && /tmp/from-pypi/bin/pip install warplens==<version>
   ```

9. Write the date into the `CHANGELOG.md` heading if step 2 left it open, and make the GitHub
   release for the tag from that section.

## What is in the two files

- The wheel is pure Python (`py3-none-any`): the `warplens` package, the two machine models
  (`warplens/configs/*.yaml`), and the shim's C sources and build scripts
  (`warplens/shim/`). The shim libraries are not built into it: `warplens run` compiles them
  on the user's machine on first use, into `~/.cache/warplens`, with the C compiler and the
  CUDA headers found there. The data files are listed under `[tool.setuptools.package-data]`
  in `pyproject.toml`; a new one that the package reads at run time is added there.
- The sdist is the whole checked-in tree, tests, examples, validation programs and documents
  included (`MANIFEST.in`), so the suite runs from an unpacked sdist. A new top-level
  directory is added to `MANIFEST.in`.

`tests/packaging/test_packaging.py` builds both and fails when a file of the package is missing from
the wheel, when the sdist is not the tree that git has, or when a build script has lost its
executable bit. It needs `build`, which the `dev` extra installs, and is skipped without it.
