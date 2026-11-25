# The GEMM kernels of *Modern GPU Programming for MLSys*

The [book](https://mlc.ai/modern-gpu-programming-for-mlsys/) develops a GEMM
for Blackwell in nine steps, `hgemm_v1` to `hgemm_v9`, written in TIRx. This
directory runs them in the simulator, from your own clone of the book: its
repository has no license, so none of its code is here.

| File | What it is |
| --- | --- |
| `run.py` | reads the kernel definitions from the chapters of your clone, compiles each one and checks its result against NumPy |
| `book_tour.ipynb` | a notebook that follows the book's GEMM chapters and shows, for each kernel, what the simulator sees: the PTX it became, which warp does what, the findings, the tensor-memory layout, the timing model's account |
| `booksim.py` | the notebook's helpers. They use `run.py` and warplens's public API (`warplens.run_program`) and nothing else |

## What you need

- warplens installed from a checkout, on Linux, with a C compiler and the
  CUDA toolkit headers (`warplens run` builds its shim libraries from them);
- `pip install -e ".[tirx]"` (`apache-tvm-tirx` and `cuda-bindings`), and an
  NVRTC library that knows `sm_100a` (CUDA 12.8 or newer; the repository's
  conda env has one, see `env/README.md`);
- a clone of the book:

  ```bash
  git clone https://github.com/mlc-ai/modern-gpu-programming-for-mlsys book
  export WARPLENS_BOOK=$PWD/book
  ```

No GPU is used. The tests run against the book's commit `3a24144d`
(`.github/workflows/ci.yml`); a later one may have changed the chapters.

## The kernels

From the top of the checkout:

```bash
warplens run python examples/08_book/run.py "$WARPLENS_BOOK"          # hgemm_v1 ... hgemm_v9
warplens run python examples/08_book/run.py "$WARPLENS_BOOK" v1 v7    # only these
```

Each kernel gets the smallest problem it does any work on. All nine take
about a minute and a half, most of it compilation. `hgemm_v7` to `v9` get one
finding each, `tmem-dealloc-repeated`, which section 6.4 of the
[user guide](../../docs/guide.md) explains; section 5.6 has the rest. Without
`warplens run`, the same command runs the kernels on a Blackwell GPU.

## The notebook

Jupyter is optional. With it:

```bash
pip install -e ".[jupyter]"       # notebook, nbclient, nbformat, ipykernel, and the tirx extra
jupyter notebook examples/08_book/book_tour.ipynb
```

The notebook finds the book through `WARPLENS_BOOK`, or edit the path in its
first code cell. It keeps the outputs of a run, so it can be read without
running anything.

Without Jupyter, the notebook still runs: its code cells are plain Python (no
`%` or `!` lines), and joined together they are a script. Pictures are then
replaced by a line that says what they would show.

```bash
python -c 'import json, sys
cells = json.load(open(sys.argv[1]))["cells"]
print("\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"))
' examples/08_book/book_tour.ipynb > /tmp/book_tour.py
python /tmp/book_tour.py          # from the top of the checkout, or from this directory
```

`tests/documentation/test_guide.py` executes the notebook when `WARPLENS_BOOK` is set:
through `nbclient` when the Jupyter packages are installed, and as that
script when they are not.
