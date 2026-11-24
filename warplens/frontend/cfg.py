"""Control-flow graph and reconvergence points for one function.

A warp whose lanes disagree at a branch splits, and the lanes rejoin at the
branch's immediate post-dominator: the first instruction every path from the
branch must pass through.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Function, Instr


def _is_terminator(ins: Instr) -> bool:
    return ins.opcode in ("bra", "ret", "exit")


@dataclass(slots=True)
class CFG:
    starts: list[int]  # first pc of each block
    succs: list[list[int]]  # block -> successor blocks; len(starts) is the virtual exit
    block_of: list[int]  # pc -> block

    @property
    def exit(self) -> int:
        return len(self.starts)


def build_cfg(fn: Function) -> CFG:
    n = len(fn.instrs)
    leaders = {0} | {pc for pc in fn.labels.values() if pc < n}
    for pc, ins in enumerate(fn.instrs):
        if _is_terminator(ins) and pc + 1 < n:
            leaders.add(pc + 1)
    starts = sorted(pc for pc in leaders if pc < n)
    block_at = {pc: b for b, pc in enumerate(starts)}
    exit_block = len(starts)

    block_of = []
    for b, start in enumerate(starts):
        end = starts[b + 1] if b + 1 < len(starts) else n
        block_of.extend([b] * (end - start))

    def block_for(pc: int) -> int:
        return block_at[pc] if pc < n else exit_block

    succs: list[list[int]] = []
    for b, start in enumerate(starts):
        end = starts[b + 1] if b + 1 < len(starts) else n
        last = fn.instrs[end - 1]
        out: list[int] = []
        if last.opcode == "bra":
            out.append(block_for(fn.labels[last.operands[0].name]))
            if last.pred is not None:
                out.append(block_for(end))
        elif last.opcode in ("ret", "exit"):
            out.append(exit_block)
            if last.pred is not None:
                out.append(block_for(end))
        else:
            out.append(block_for(end))
        succs.append(list(dict.fromkeys(out)))
    return CFG(starts, succs, block_of)


def _reaching(succs: list[list[int]], exit_node: int) -> list[bool]:
    """Which blocks have some path to the exit."""
    preds: list[list[int]] = [[] for _ in range(exit_node + 1)]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)
    seen = [False] * (exit_node + 1)
    seen[exit_node] = True
    work = [exit_node]
    while work:
        for p in preds[work.pop()]:
            if not seen[p]:
                seen[p] = True
                work.append(p)
    return seen


def immediate_post_dominators(succs: list[list[int]], exit_node: int) -> list[int]:
    """Cooper-Harvey-Kennedy on the reversed graph."""
    n = exit_node + 1
    reaches_exit = _reaching(succs, exit_node)
    # Give blocks stuck in an infinite loop a virtual edge to the exit, so that
    # nothing is claimed to post-dominate a branch into such a loop.
    succs = [out if reaches_exit[b] else [*out, exit_node] for b, out in enumerate(succs)]
    preds: list[list[int]] = [[] for _ in range(n)]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)

    # Postorder of a DFS from the exit along reversed edges.
    order: list[int] = []
    seen = [False] * n
    seen[exit_node] = True
    stack = [(exit_node, iter(preds[exit_node]))]
    while stack:
        node, it = stack[-1]
        for nxt in it:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append((nxt, iter(preds[nxt])))
                break
        else:
            order.append(node)
            stack.pop()
    rank = {node: i for i, node in enumerate(order)}  # exit has the highest rank

    ipdom: dict[int, int] = {exit_node: exit_node}

    def intersect(a: int, b: int) -> int:
        while a != b:
            while rank[a] < rank[b]:
                a = ipdom[a]
            while rank[b] < rank[a]:
                b = ipdom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in reversed(order[:-1]):
            new = None
            for s in succs[b]:
                if s in ipdom:
                    new = s if new is None else intersect(new, s)
            if new is not None and ipdom.get(b) != new:
                ipdom[b] = new
                changed = True
    return [ipdom.get(b, exit_node) for b in range(exit_node)]


def block_reachability(cfg: CFG) -> list[int]:
    """For every block, the set of blocks that some path of one or more edges leads to from
    it, as a bit set (bit b for block b). The virtual exit has the last entry and reaches
    nothing; a block is in its own set only when it is on a cycle."""
    n = cfg.exit
    reach = [0] * (n + 1)
    changed = True
    while changed:
        changed = False
        for b in range(n - 1, -1, -1):
            new = reach[b]
            for s in cfg.succs[b]:
                if s != n:
                    new |= (1 << s) | reach[s]
            if new != reach[b]:
                reach[b] = new
                changed = True
    return reach


def reconvergence_points(fn: Function) -> dict[int, int]:
    """Map the pc of every predicated branch to the pc where its lanes rejoin.
    A value of len(fn.instrs) means they only rejoin by leaving the function."""
    n = len(fn.instrs)
    if n == 0:
        return {}
    cfg = build_cfg(fn)
    ipdom = immediate_post_dominators(cfg.succs, cfg.exit)
    out: dict[int, int] = {}
    for pc, ins in enumerate(fn.instrs):
        if ins.opcode == "bra" and ins.pred is not None:
            r = ipdom[cfg.block_of[pc]]
            out[pc] = cfg.starts[r] if r != cfg.exit else n
    return out
