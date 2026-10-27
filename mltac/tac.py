"""Three-address code (TAC) IR, interpreter, cost model and optimization passes.

An instruction is a tuple (dst, op, a, b):
    copy : dst = a                (b is None)
    + - * // << >> : dst = a op b
Operands are either variable names (str) or integer constants (int).
A Program has live-in variables (inputs) and live-out variables (outputs).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

Operand = Union[str, int]
Instr = Tuple[str, str, Operand, Optional[Operand]]

COMMUTATIVE = {"+", "*"}
# Simple latency-style cost model (relative cycles).
COST = {"copy": 1, "+": 1, "-": 1, "<<": 1, ">>": 1, "*": 3, "//": 8}


@dataclass
class Program:
    instrs: List[Instr]
    inputs: List[str]
    outputs: List[str]

    def copy(self) -> "Program":
        return Program(list(self.instrs), list(self.inputs), list(self.outputs))

    def text(self) -> str:
        lines = []
        for d, op, a, b in self.instrs:
            lines.append(f"{d} = {a}" if op == "copy" else f"{d} = {a} {op} {b}")
        return "\n".join(lines)

    def key(self) -> str:
        return "|".join(map(str, self.instrs)) + "#" + ",".join(self.outputs)


def cost(p: Program) -> int:
    return sum(COST[op] for _, op, _, _ in p.instrs)


def is_const(x) -> bool:
    return isinstance(x, int)


def eval_op(op: str, a: int, b: Optional[int]) -> int:
    if op == "copy":
        return a
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "//":
        return a // b
    if op == "<<":
        return a << b
    if op == ">>":
        return a >> b
    raise ValueError(op)


def run(p: Program, env_in: dict) -> dict:
    env = dict(env_in)
    for d, op, a, b in p.instrs:
        av = a if is_const(a) else env[a]
        bv = None if b is None else (b if is_const(b) else env[b])
        env[d] = eval_op(op, av, bv)
    return {o: env[o] for o in p.outputs}


def equivalent(p: Program, q: Program, trials: int = 20, seed: int = 0) -> bool:
    """Semantic-equivalence check by random testing on the live-in variables."""
    import random
    rng = random.Random(seed)
    for _ in range(trials):
        env = {v: rng.randint(-50, 50) for v in p.inputs}
        if run(p, env) != run(q, env):
            return False
    return True


# ----------------------------------------------------------------- passes
def _val(x, env):
    return env.get(x, x) if isinstance(x, str) else x


def constant_fold(p: Program) -> Program:
    """Constant propagation + folding (forward, handles re-assignment)."""
    consts = {}
    out = []
    for d, op, a, b in p.instrs:
        a2 = consts.get(a, a) if isinstance(a, str) else a
        b2 = consts.get(b, b) if isinstance(b, str) else b
        if op == "copy":
            if is_const(a2):
                consts[d] = a2
            else:
                consts.pop(d, None)
            out.append((d, "copy", a2, None))
            continue
        if is_const(a2) and is_const(b2) and not (op == "//" and b2 == 0):
            v = eval_op(op, a2, b2)
            consts[d] = v
            out.append((d, "copy", v, None))
        else:
            consts.pop(d, None)
            out.append((d, op, a2, b2))
    return Program(out, p.inputs, p.outputs)


def _pow2(n) -> int:
    return is_const(n) and n > 0 and (n & (n - 1)) == 0


def algebraic_simplify(p: Program) -> Program:
    """Algebraic identities and strength reduction."""
    out = []
    for d, op, a, b in p.instrs:
        if op == "copy":
            out.append((d, op, a, b)); continue
        # canonicalise: constant on the right for commutative ops
        if op in COMMUTATIVE and is_const(a) and not is_const(b):
            a, b = b, a
        new = (d, op, a, b)
        if op == "+" and b == 0:
            new = (d, "copy", a, None)
        elif op == "-" and b == 0:
            new = (d, "copy", a, None)
        elif op == "-" and a == b and isinstance(a, str):
            new = (d, "copy", 0, None)
        elif op == "*" and b == 1:
            new = (d, "copy", a, None)
        elif op == "*" and b == 0:
            new = (d, "copy", 0, None)
        elif op == "*" and _pow2(b):
            k = b.bit_length() - 1
            new = (d, "+", a, a) if k == 1 else (d, "<<", a, k)
        elif op == "//" and b == 1:
            new = (d, "copy", a, None)
        elif op == "//" and _pow2(b):
            new = (d, ">>", a, b.bit_length() - 1)
        out.append(new)
    return Program(out, p.inputs, p.outputs)


def cse(p: Program) -> Program:
    """Common subexpression elimination via available expressions."""
    avail = {}  # (op,a,b) -> var holding the value
    out = []
    for d, op, a, b in p.instrs:
        if op == "copy":
            key = None
        else:
            x, y = (a, b)
            if op in COMMUTATIVE and str(x) > str(y):
                x, y = y, x
            key = (op, x, y)
        if key is not None and key in avail and avail[key] != d:
            out.append((d, "copy", avail[key], None))
        else:
            out.append((d, op, a, b))
        # kill expressions that depended on / were held in d
        avail = {k: v for k, v in avail.items()
                 if v != d and d not in (k[1], k[2])}
        if key is not None and d not in (a, b) and key not in avail:
            avail[key] = d
    return Program(out, p.inputs, p.outputs)


def copy_propagate(p: Program) -> Program:
    copies = {}
    out = []
    for d, op, a, b in p.instrs:
        a2 = copies.get(a, a) if isinstance(a, str) else a
        b2 = copies.get(b, b) if isinstance(b, str) else b
        out.append((d, op, a2, b2))
        copies = {k: v for k, v in copies.items() if k != d and v != d}
        if op == "copy" and a2 != d:
            copies[d] = a2
    return Program(out, p.inputs, p.outputs)


def dead_code_elim(p: Program) -> Program:
    live = set(p.outputs)
    keep = []
    for d, op, a, b in reversed(p.instrs):
        if d in live:
            keep.append((d, op, a, b))
            live.discard(d)
            for x in (a, b):
                if isinstance(x, str):
                    live.add(x)
    keep.reverse()
    return Program(keep, p.inputs, p.outputs)


PASSES = {
    "cf": constant_fold,
    "alg": algebraic_simplify,
    "cse": cse,
    "cp": copy_propagate,
    "dce": dead_code_elim,
}

# Candidate pass sequences = the label space the ML model chooses from.
SEQUENCES = {
    "S0_none":        [],
    "S1_dce":         ["dce"],
    "S2_cf":          ["cf", "dce"],
    "S3_alg":         ["alg", "dce"],
    "S4_cse":         ["cse", "cp", "dce"],
    "S5_cf_alg":      ["cf", "alg", "dce"],
    "S6_cf_cse":      ["cf", "cse", "cp", "dce"],
    "S7_alg_cse":     ["alg", "cse", "cp", "dce"],
    "S8_full":        ["cf", "alg", "cse", "cp", "dce"],
    "S9_full_x2":     ["cf", "alg", "cse", "cp", "dce", "cf", "alg", "cp", "dce"],
}
FULL_PIPELINE = "S9_full_x2"   # the 'fixed -O2-style' baseline


def apply_sequence(p: Program, seq_name: str) -> Program:
    q = p
    for name in SEQUENCES[seq_name]:
        q = PASSES[name](q)
    return q


def compile_effort(seq_name: str) -> int:
    """Proxy for compile time: number of pass invocations."""
    return len(SEQUENCES[seq_name])
