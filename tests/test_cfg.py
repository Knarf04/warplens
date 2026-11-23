from warplens.frontend.cfg import reconvergence_points

from conftest import kernel

REGS = ".reg .pred %p; .reg .b32 %r;"


def reconv_labels(body):
    fn = kernel(REGS + body).functions["k"]
    names = {pc: name for name, pc in fn.labels.items()}
    names[len(fn.instrs)] = "<exit>"
    return [names[pc] for _, pc in sorted(reconvergence_points(fn).items())]


def test_if_else_rejoins_after_both_arms():
    assert reconv_labels("""
        @%p bra ELSE;
        mov.u32 %r, 1;
        bra JOIN;
      ELSE:
        mov.u32 %r, 2;
      JOIN:
        ret;
    """) == ["JOIN"]


def test_loop_branch_rejoins_at_loop_exit():
    assert reconv_labels("""
      HEAD:
        add.u32 %r, %r, 1;
        @%p bra HEAD;
      AFTER:
        ret;
    """) == ["AFTER"]


def test_early_return_only_rejoins_at_exit():
    assert reconv_labels("""
        @%p bra SKIP;
        ret;
      SKIP:
        mov.u32 %r, 1;
        ret;
    """) == ["<exit>"]


def test_break_inside_loop_with_nested_if():
    # Same shape as the nvcc output for examples/00_divergent `classify`.
    assert reconv_labels("""
      HEAD:
        @%p bra OUT;
        @%p bra B;
        mov.u32 %r, 1;
        bra LATCH;
      B:
        mov.u32 %r, 2;
      LATCH:
        @%p bra HEAD;
      OUT:
        ret;
    """) == ["OUT", "LATCH", "OUT"]


def test_infinite_loop_does_not_hang_analysis():
    assert reconv_labels("""
        @%p bra SPIN;
        ret;
      SPIN:
        bra SPIN;
    """) == ["<exit>"]
