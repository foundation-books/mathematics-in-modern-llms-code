"""Instructor checks: exact fixtures, graph semantics, and finite diagnostics."""
import argparse
import json
import sys
from pathlib import Path

import sympy as sp

from homework_starter import (
    A, B, K0, Q, is_symbolic_zero, marginal_residual_vectors,
    require_positive_square, sinkhorn_sweep, x0,
)
from chapter11_fixture import check_fixtures


def assert_zero(value):
    entries = list(value) if isinstance(value, sp.MatrixBase) else [value]
    assert all(is_symbolic_zero(e) for e in entries), value


def sigmoid(t):
    return 1 / (1 + sp.exp(-t))


def C(p):
    return sp.Matrix([[p, 1 - p], [1 - p, p]])


def mean_read(sources):
    assert sources
    return sum(sources, sp.zeros(sources[0].rows, 1)) / len(sources)


def full_history(branches, start=x0):
    sources = [sp.Matrix(start)]
    trace = []
    for branch in branches:
        trace.append([v.copy() for v in sources])
        sources.append(branch(mean_read(sources)))
    return mean_read(sources), trace


def block_history(branches, block_size, start=x0):
    if not isinstance(block_size, int) or block_size < 1 or not branches:
        raise ValueError("Need branches and a positive integer block size.")
    completed = []
    partial = sp.zeros(start.rows, 1)
    partial_count = 0
    trace = []
    for ell, branch in enumerate(branches):
        sources = [start, *completed]
        # Existence is independent of the numerical value of this sum.
        if partial_count:
            sources.append(partial)
        trace.append([v.copy() for v in sources])
        partial += branch(mean_read(sources))
        partial_count += 1
        if partial_count == block_size or ell == len(branches) - 1:
            completed.append(partial)
            partial = sp.zeros(start.rows, 1)
            partial_count = 0
    return mean_read([start, *completed]), trace, completed


def checks_A():
    x1, x2 = check_fixtures()
    xs = sp.Matrix(sp.symbols("u v", real=True))
    graph = xs + A * xs
    graph = graph + B * graph
    product = (sp.eye(2) + B) * (sp.eye(2) + A)
    assert_zero(graph.jacobian(xs) - product)
    assert graph.subs(dict(zip(xs, x0))) == x2
    reverse = (sp.eye(2) + A) * (sp.eye(2) + B)
    assert reverse != product
    assert product.det() == 1 and product.rank() == 2
    assert (B * A).rank() == 1
    for depth in (1, 2, 5):
        for slope, expected in [(1, 2**depth), (-Q(1, 2), Q(1, 2)**depth), (-1, 0)]:
            assert (1 + slope)**depth == expected
    return {"x1": str(x1), "x2": str(x2), "jacobian": str(product),
            "reverse_discrepancy": str(product - reverse), "ranks": [2, 1]}


def checks_B():
    xs = sp.Matrix(sp.symbols("x1 x2", real=True))
    ones = sp.ones(2, 1)
    f = A * xs
    gate = xs.applyfunc(sigmoid)
    dg = gate.jacobian(xs)
    additive = xs + gate.multiply_elementwise(f)
    interpolation = (ones - gate).multiply_elementwise(xs) + gate.multiply_elementwise(f)
    hand_add = sp.eye(2) + sp.diag(*gate) * A + sp.diag(*f) * dg
    hand_interp = sp.diag(*(ones - gate)) + sp.diag(*gate) * A + sp.diag(*(f - xs)) * dg
    assert_zero(additive.jacobian(xs) - hand_add)
    assert_zero(interpolation.jacobian(xs) - hand_interp)
    gamma = sp.symbols("gamma", real=True)
    assert_zero((xs + gamma * f).jacobian(xs) - (sp.eye(2) + gamma * A))
    assert_zero(((1 - gamma) * xs + gamma * f).jacobian(xs)
                - ((1 - gamma) * sp.eye(2) + gamma * A))
    x = sp.symbols("x", real=True)
    assert sp.diff(x + sp.tanh(x), x).subs(x, 0) == 2
    assert_zero((xs + gamma * f).diff(gamma) - f)
    return {"complete_gate_identities": "passed", "zero_gate_derivative": 2}


def checks_C():
    r0 = A * x0
    r1 = B * (x0 + r0)
    frozen = x0 / 6 + r0 / 3 + r1 / 2
    assert frozen == sp.Matrix([Q(1, 4), -Q(1, 24)])
    scores = sp.symbols("s0:3", real=True)
    tau = sp.symbols("tau", positive=True)
    exps = [sp.exp(s / tau) for s in scores]
    alpha = [e / sum(exps) for e in exps]
    values = [x0, r0, r1]
    h = sum((a * v for a, v in zip(alpha, values)), sp.zeros(2, 1))
    for i in range(3):
        assert_zero(h.diff(scores[i]) - alpha[i] * (values[i] - h) / tau)
    branches = [lambda h: A * h, lambda h: B * h]
    full, ft = full_history(branches)
    block, bt, _ = block_history(branches, 2)
    assert full == sp.Matrix([Q(13, 36), Q(47, 216)])
    assert block == sp.Matrix([Q(13, 24), Q(47, 144)])
    assert ft == bt and [len(v) for v in ft] == [1, 2]
    bad_initial_read = mean_read([x0, sp.zeros(2, 1)])
    assert bad_initial_read == x0 / 2 and bad_initial_read != x0
    e1 = sp.Matrix([1, 0])
    cancelling = [lambda h: e1, lambda h: -e1, lambda h: sp.zeros(2, 1)]
    _, ct, _ = block_history(cancelling, 3)
    assert len(ct[2]) == 2 and ct[2][1] == sp.zeros(2, 1)
    assert mean_read(ct[2]) == x0 / 2  # a real zero source stays present
    longer = [*branches, lambda h: A * h]
    full_long, fl = full_history(longer)
    unit_block, ut, _ = block_history(longer, 1)
    assert unit_block == full_long and ut == fl
    grouped_long, gt, completed = block_history(longer, 2)
    assert len(completed) == 2  # last block is shorter, not discarded
    assert [len(v) for v in gt] == [1, 2, 2]
    assert grouped_long != full_long and gt[2] != fl[2]
    pair1 = [sp.Matrix([1, 0]), sp.Matrix([0, 1])]
    pair2 = [sp.Matrix([1, 1]), sp.zeros(2, 1)]
    assert sum(pair1, sp.zeros(2, 1)) == sum(pair2, sp.zeros(2, 1))
    assert 3 * pair1[0] / 4 + pair1[1] / 4 != 3 * pair2[0] / 4 + pair2[1] / 4
    return {"frozen": str(frozen), "full": str(full), "block": str(block),
            "source_counts": [len(v) for v in ft], "long_block_counts": [len(v) for v in gt],
            "absent_and_cancelling_zero_sources": "passed"}


def checks_D():
    x1 = x0 + A * x0
    H = sp.Matrix.vstack(x0.T, x1.T)
    a = sp.Matrix([Q(1, 2), Q(1, 2)])
    b = sp.ones(2, 1)
    M = C(Q(3, 4))
    u = H.T * a
    z = A * u
    new = M * H + b * z.T
    assert u == sp.Matrix([Q(7, 8), Q(5, 6)])
    assert new == sp.Matrix([[Q(19, 16), Q(5, 6)], [Q(67, 48), Q(5, 6)]])
    ones = sp.ones(2, 1)
    assert ones.T * M * H / 2 == ones.T * H / 2
    assert ones.T * new / 2 == sp.Matrix([[Q(31, 24), Q(5, 6)]])
    assert ones.T * new / 2 - ones.T * H / 2 == z.T
    p = sp.symbols("p", real=True)  # admissibility still needs 0 <= p <= 1
    P = ones * ones.T / 2
    disagreement = sp.Matrix([1, -1])
    assert C(p) * ones == ones and ones.T * C(p) == ones.T
    assert_zero(C(p) * disagreement - (2 * p - 1) * disagreement)
    for n in range(1, 7):
        assert_zero(C(p)**n - (P + (2 * p - 1)**n * (sp.eye(2) - P)))
    assert [C(p0).rank() for p0 in [0, Q(1, 2), Q(3, 4), 1]] == [2, 1, 2, 2]
    assert C(2) * disagreement == 3 * disagreement
    R = sp.Matrix([[1, 0], [1, 0]])
    e1 = sp.Matrix([1, 0])
    assert R * e1 == ones and (R * e1).dot(R * e1) == 2
    assert ones.T * R * e1 / 2 != ones.T * e1 / 2
    hs = sp.Matrix(2, 2, sp.symbols("h0:4", real=True))
    es = sp.Matrix(2, 2, sp.symbols("e0:4", real=True))
    eta = sp.symbols("eta", real=True)
    fixed_update = M * (hs + eta * es) + b * (A * (hs + eta * es).T * a).T
    assert_zero(fixed_update.diff(eta) - (M * es + b * (A * es.T * a).T))
    return {"u": str(u), "updated_streams": str(new), "injected_mean": str(z.T),
            "checked_integer_powers": list(range(1, 7)), "rank_loss_and_signed_witnesses": "passed"}


def marginal_errors(M):
    rows, cols = marginal_residual_vectors(M)
    return max(abs(e) for e in rows), max(abs(e) for e in cols)


def checks_E():
    before = K0.copy()
    first = sinkhorn_sweep(K0)
    second = sinkhorn_sweep(first)
    assert K0 == before  # helper does not mutate its caller
    assert first == sp.Matrix([[Q(3, 7), Q(4, 7)], [Q(9, 17), Q(8, 17)]])
    assert marginal_errors(first) == (0, Q(5, 119))
    assert second == sp.Matrix([[Q(31, 69), Q(38, 69)], [Q(93, 169), Q(76, 169)]])
    assert marginal_errors(second) == (0, Q(5, 11661))
    cross = lambda K: K[0, 0] * K[1, 1] / (K[0, 1] * K[1, 0])
    assert cross(K0) == cross(first) == cross(second) == Q(2, 3)
    pstar = 2 / (2 + sp.sqrt(6))
    limit = C(pstar)
    assert_zero(cross(limit) - cross(K0))
    assert_zero(limit * sp.ones(2, 1) - sp.ones(2, 1))
    assert_zero(sp.ones(1, 2) * limit - sp.ones(1, 2))
    bound = sp.sqrt(1 + Q(5, 119))
    # Independently evaluate the largest singular value of this finite matrix.
    spectral = sp.sqrt(max((first.T * first).eigenvals()))
    assert float(spectral) <= float(bound)
    rejected = [sp.zeros(0, 0), sp.ones(2, 3), sp.eye(2),
                sp.Matrix([[-1, 1], [1, 1]]),
                sp.Matrix([[sp.symbols("unknown", real=True)]])]
    for bad in rejected:
        try:
            sinkhorn_sweep(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid scaling input accepted: {bad}")
    positive = sp.symbols("positive", positive=True)
    require_positive_square(sp.Matrix([[positive]]))
    assert sinkhorn_sweep(sp.Matrix([[positive]])) == sp.ones(1, 1)
    for bad in [sp.zeros(0, 0), sp.ones(2, 3)]:
        try:
            marginal_residual_vectors(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid residual dimensions accepted")
    current = K0.evalf(80)
    numerical_limit = limit.evalf(80)
    table = []
    for sweep in range(1, 21):
        current = sinkhorn_sweep(current)
        if sweep in (1, 2, 5, 10, 20):
            er, ec = marginal_errors(current)
            distance = max(abs(e) for e in current - numerical_limit)
            table.append({"sweeps": sweep, "row_error": str(er.evalf(12)),
                          "column_error": str(ec.evalf(12)),
                          "max_entry_distance": str(distance.evalf(12))})
    er, ec = marginal_errors(current)
    tol = sp.Float("1e-35", 80)
    assert er < tol and ec < tol
    assert max(abs(e) for e in current - numerical_limit) < tol
    return {"first": str(first), "second": str(second), "pstar": str(pstar),
            "order": "column-then-row", "decimal_precision": 80,
            "first_norm_bound": str(bound), "numerical_table": table,
            "checked_rejected_inputs": len(rejected)}


def checks_F():
    x, y = sp.symbols("x y", real=True)
    h = sp.Matrix([x, y])
    delta = x - y
    p = sigmoid(2 * delta)
    mapping = C(p) * h
    jacobian = mapping.jacobian(h)
    dp = sp.Matrix([[sp.diff(p, x), sp.diff(p, y)]])
    omitted = sp.Matrix([delta, -delta]) * dp
    assert_zero(jacobian - C(p) - omitted)
    assert_zero(mapping[0] - mapping[1] - delta * sp.tanh(delta))
    point = {x: Q(1, 2), y: -Q(1, 2)}
    full = jacobian.subs(point)
    frozen = C(p).subs(point)
    t = sp.tanh(1)
    gain = t + 1 - t**2
    assert_zero(gain - (1 + t * (1 - t)))
    assert float(gain) > 1
    v = sp.Matrix([1, -1]) / sp.sqrt(2)
    assert_zero(full * v - gain * v)
    assert_zero(frozen * v - t * v)
    assert_zero(full * sp.ones(2, 1) - sp.ones(2, 1))
    step = sp.Rational(1, 10**25)
    hstar = sp.Matrix([Q(1, 2), -Q(1, 2)])
    plus = mapping.subs(dict(zip(h, hstar + step * v))).evalf(80)
    minus = mapping.subs(dict(zip(h, hstar - step * v))).evalf(80)
    numerical = (plus - minus) / (2 * step)
    discrepancy = max(abs(e) for e in numerical - (gain * v).evalf(80))
    assert discrepancy < sp.Float("1e-45", 80)
    return {"full_jacobian": str(full.applyfunc(sp.simplify)),
            "frozen_jacobian": str(frozen), "gain": str(gain.evalf(20)),
            "frozen_gain": str(t.evalf(20)), "directional_difference_error": str(discrepancy.evalf(8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write verification output outside the repository")
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "sympy": sp.__version__}
    for label, check in zip("ABCDEF", [checks_A, checks_B, checks_C, checks_D, checks_E, checks_F]):
        report[label] = check()
        print(f"Part {label}: passed")
    print(f"Python {report['python']}; SymPy {report['sympy']}")
    for row in report["E"]["numerical_table"]:
        print("Sinkhorn:", row)
    print("Dynamic local gain:", report["F"]["gain"])
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print("Verification record:", args.json)


if __name__ == "__main__":
    main()
