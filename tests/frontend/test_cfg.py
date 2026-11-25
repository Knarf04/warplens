from warplens.frontend.cfg import block_reachability, build_cfg, reconvergence_points

from tests.support import kernel

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


def test_block_reachability():
    fn = kernel(REGS + """
        @%p bra B;
      A:
        @%p bra A;
        ret;
      B:
        mov.u32 %r, 1;
        ret;
    """).functions["k"]
    cfg = build_cfg(fn)
    entry, a, a_ret, b = range(4)
    reach = block_reachability(cfg)
    assert reach[entry] == 1 << a | 1 << a_ret | 1 << b
    assert reach[a] == 1 << a | 1 << a_ret  # on a cycle: reaches itself
    assert reach[a_ret] == 0 and reach[b] == 0 and reach[cfg.exit] == 0


def test_behind_is_a_strict_order_outside_cycles():
    from warplens.core.program import Program
    module = kernel(REGS + """
        @%p bra ELSE;
        mov.u32 %r, 1;
      JOIN:
        mov.u32 %r, 2;
        ret;
      ELSE:
        @%p bra JOIN;
      LOOP:
        mov.u32 %r, 3;
        @%p bra LOOP;
        ret;
    """)
    prog = Program(module, module.functions["k"])
    then, join, els, loop = 1, prog.fn.labels["JOIN"], prog.fn.labels["ELSE"], prog.fn.labels["LOOP"]
    assert prog.behind(then, join) and not prog.behind(join, then)
    assert prog.behind(els, join) and not prog.behind(join, els)
    assert not prog.behind(then, els) and not prog.behind(els, then)  # neither reaches the other
    assert prog.behind(join, join + 1) and not prog.behind(join + 1, join)  # within a block
    assert not prog.behind(loop, loop + 1) and not prog.behind(loop + 1, loop)  # on one cycle
