"""Synthetic program generator, static feature extraction, and a tiny front end."""
import ast
import random
from typing import List

from .tac import Program, dead_code_elim, is_const, _pow2

INPUTS = ["a", "b", "c", "d"]
BIN_OPS = ["+", "-", "*"]


# ------------------------------------------------------------- generator
def random_profile(rng: random.Random) -> dict:
    """Each program gets its own 'personality' so different passes win."""
    return {
        "p_const": rng.choice([0.0, 0.05, 0.2, 0.4]),
        "p_repeat": rng.choice([0.0, 0.05, 0.2, 0.4]),
        "p_pow2": rng.choice([0.0, 0.05, 0.2, 0.4]),
        "p_ident": rng.choice([0.0, 0.05, 0.15]),
        "p_copy": rng.choice([0.0, 0.1, 0.25]),
        "p_dead": rng.choice([0.0, 0.1, 0.3]),
        "p_reassign": rng.choice([0.0, 0.1, 0.25]),
    }


def generate_program(rng: random.Random, n_lo=8, n_hi=40, prof=None) -> Program:
    prof = prof or random_profile(rng)
    n = rng.randint(n_lo, n_hi)
    defined = list(INPUTS)
    instrs, exprs, temps = [], [], 0

    def pick():
        if len(defined) > 6 and rng.random() < 0.7:
            return rng.choice(defined[-5:])      # mostly use recent values
        return rng.choice(defined)

    for _ in range(n):
        is_dead = rng.random() < prof["p_dead"] * 0.35
        if is_dead:
            dst = f"u{temps}"; temps += 1            # never read, never an output
        elif rng.random() < prof["p_reassign"] and len(defined) > 4:
            dst = rng.choice([d for d in defined[4:] if not d.startswith("u")] or defined[:1])
        else:
            dst = f"t{temps}"; temps += 1
        r = rng.random()
        acc = 0.0
        def hit(p):
            nonlocal acc
            acc += p
            return r < acc
        if exprs and hit(prof["p_repeat"]):
            _, op, a, b = rng.choice(exprs)
            ins = (dst, op, a, b)
        elif hit(prof["p_const"]):
            op = rng.choice(BIN_OPS)
            ins = (dst, op, rng.randint(1, 9), rng.randint(1, 9)) if rng.random() < .6 \
                else (dst, op, pick(), rng.randint(1, 9))
        elif hit(prof["p_pow2"]):
            if rng.random() < .75:
                ins = (dst, "*", pick(), rng.choice([2, 4, 8, 16]))
            else:
                ins = (dst, "//", pick(), rng.choice([2, 4, 8]))
        elif hit(prof["p_ident"]):
            ins = rng.choice([(dst, "+", pick(), 0), (dst, "*", pick(), 1),
                              (dst, "-", pick(), 0), (dst, "*", pick(), 0)])
        elif hit(prof["p_copy"]):
            ins = (dst, "copy", pick(), None)
        else:
            ins = (dst, rng.choice(BIN_OPS), pick(), pick())
        instrs.append(ins)
        if ins[1] != "copy":
            exprs.append(ins)
        if dst not in defined and not dst.startswith("u"):
            defined.append(dst)
    # outputs = values that are never read after their last definition (sinks),
    # excluding deliberately-dead 'u' variables; cap at 6 (extras become dead).
    last_def = {}
    for i, (d, *_) in enumerate(instrs):
        last_def[d] = i
    sinks = []
    for d, i in last_def.items():
        if d.startswith("u"):
            continue
        used_after = any(d in (a, b) for _, _, a, b in instrs[i + 1:])
        if not used_after:
            sinks.append((i, d))
    sinks.sort()
    outs = [d for _, d in sinks][-6:] or [instrs[-1][0]]
    return Program(instrs, list(INPUTS), outs)


# -------------------------------------------------------------- features
FEATURE_NAMES = [
    "n_instr", "n_add", "n_sub", "n_mul", "n_div", "n_copy",
    "n_const_const", "n_const_operand", "n_dup_expr", "n_identity",
    "n_pow2_mul", "n_pow2_div", "n_dead", "n_reassign", "n_outputs",
    "n_unique_vars", "frac_const", "frac_dup", "frac_dead", "frac_pow2",
    "frac_copy", "frac_identity",
]


def extract_features(p: Program) -> List[float]:
    ins = p.instrs
    n = max(len(ins), 1)
    ops = [i[1] for i in ins]
    n_cc = sum(1 for _, op, a, b in ins if op != "copy" and is_const(a) and is_const(b))
    n_c1 = sum(1 for _, op, a, b in ins if is_const(a) or (b is not None and is_const(b)))
    seen, dup = set(), 0
    for _, op, a, b in ins:
        if op == "copy":
            continue
        key = (op,) + tuple(sorted((str(a), str(b)))) if op in "+*" else (op, a, b)
        dup += key in seen
        seen.add(key)
    ident = sum(1 for _, op, a, b in ins
                if (op in "+-" and b == 0) or (op == "*" and b in (0, 1)) or (op == "//" and b == 1))
    p2m = sum(1 for _, op, a, b in ins if op == "*" and _pow2(b))
    p2d = sum(1 for _, op, a, b in ins if op == "//" and _pow2(b))
    dead = len(ins) - len(dead_code_elim(p).instrs)
    dsts = [d for d, *_ in ins]
    reassign = len(dsts) - len(set(dsts))
    uv = set(dsts) | {x for _, _, a, b in ins for x in (a, b) if isinstance(x, str)}
    f = [
        len(ins), ops.count("+"), ops.count("-"), ops.count("*"), ops.count("//"),
        ops.count("copy"), n_cc, n_c1, dup, ident, p2m, p2d, dead, reassign,
        len(p.outputs), len(uv),
        n_c1 / n, dup / n, dead / n, (p2m + p2d) / n, ops.count("copy") / n, ident / n,
    ]
    return [float(x) for x in f]


# -------------------------------------------------------------- frontend
class _TACBuilder(ast.NodeVisitor):
    def __init__(self):
        self.instrs, self.n = [], 0

    def tmp(self):
        self.n += 1
        return f"_t{self.n}"

    def expr(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.expr(node.operand)
            if is_const(v):
                return -v
            t = self.tmp(); self.instrs.append((t, "-", 0, v)); return t
        if isinstance(node, ast.BinOp):
            opmap = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
                     ast.Div: "//"}
            op = opmap[type(node.op)]
            a, b = self.expr(node.left), self.expr(node.right)
            t = self.tmp(); self.instrs.append((t, op, a, b)); return t
        raise SyntaxError(f"unsupported expression: {ast.dump(node)}")

    def build(self, src: str) -> Program:
        tree = ast.parse(src)
        assigned, used = [], set()
        for st in tree.body:
            if not isinstance(st, ast.Assign) or len(st.targets) != 1:
                raise SyntaxError("only simple assignments supported")
            v = self.expr(st.value)
            dst = st.targets[0].id
            self.instrs.append((dst, "copy", v, None))
            assigned.append(dst)
        names = {x for _, _, a, b in self.instrs for x in (a, b) if isinstance(x, str)}
        inputs = sorted(n for n in names if n not in {d for d, *_ in self.instrs}
                        or n in INPUTS) or ["a"]
        inputs = sorted(set(inputs) | {n for n in names if not n.startswith("_t")
                                       and n not in assigned})
        return Program(self.instrs, inputs, [assigned[-1]] if assigned else [])


def compile_source(src: str) -> Program:
    """Mini front end: Python-syntax assignments -> three-address code."""
    return _TACBuilder().build(src)
